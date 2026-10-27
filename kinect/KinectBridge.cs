// Kinect bridge: serves the Xbox 360 Kinect's colour and depth frames to the Python side over a local socket.
//
// Uses the Kinect for Windows SDK 1.8. Listens on 127.0.0.1:9601. Each request is one text line:
//   COLOR      -> a JSON header line, then width*height*4 bytes of BGRA
//   DEPTH      -> a JSON header line, then width*height*2 bytes of little-endian depth in millimetres (0 = unknown)
//   TILT <deg> -> moves the tilt motor (-27..27); answers a JSON line
//   INFO       -> a JSON line with the sensor status and the current tilt
//   AUDIO      -> a JSON header line, then an endless stream of 16 kHz mono 16-bit PCM from the microphone array
//                 (beamformed, with automatic gain and noise suppression) until the client disconnects
// The latest frame of each stream is kept, so a request never waits for the camera.
//
// kinect_bridge.exe --snapshot <folder> instead takes one colour frame (color.png) and one depth frame (depth.raw,
// 640x480 little-endian uint16 millimetres), then releases the sensor and exits. While the sensor runs, Windows gets
// silence from the Kinect's microphone array, so Hume only switches the Kinect on for the moment it looks.

using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Microsoft.Kinect;

class KinectBridge
{
    const int Port = 9601;
    static readonly object Gate = new object();
    static byte[] color;
    static short[] depth;
    static long colorStamp, depthStamp;
    static KinectSensor sensor;

    static int Main(string[] args)
    {
        sensor = KinectSensor.KinectSensors.FirstOrDefault(s => s.Status == KinectStatus.Connected);
        if (sensor == null)
        {
            Console.Error.WriteLine("No Kinect connected (check its 12 V adapter and the USB cable).");
            return 1;
        }
        sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
        sensor.DepthStream.Enable(DepthImageFormat.Resolution640x480Fps30);
        sensor.ColorFrameReady += OnColor;
        sensor.DepthFrameReady += OnDepth;
        sensor.Start();
        if (args.Length == 2 && args[0] == "--snapshot") return Snapshot(args[1]);
        Console.WriteLine("Kinect started; serving on 127.0.0.1:" + Port);

        var listener = new TcpListener(IPAddress.Loopback, Port);
        listener.Start();
        while (true)
        {
            TcpClient client = listener.AcceptTcpClient();
            new Thread(() => Serve(client)) { IsBackground = true }.Start();
        }
    }

    static int Snapshot(string folder)
    {
        // Skip the first frames: the colour camera needs a moment to set its exposure.
        DateTime until = DateTime.UtcNow.AddSeconds(8);
        long start = DateTime.UtcNow.AddMilliseconds(700).Ticks;
        while (DateTime.UtcNow < until)
        {
            lock (Gate) { if (colorStamp > start && depthStamp > start) break; }
            Thread.Sleep(30);
        }
        byte[] c; short[] d;
        lock (Gate) { c = color; d = depth; }
        sensor.Stop();
        if (c == null || d == null)
        {
            Console.Error.WriteLine("No frames from the Kinect.");
            return 2;
        }
        Directory.CreateDirectory(folder);
        using (var bmp = new System.Drawing.Bitmap(640, 480, System.Drawing.Imaging.PixelFormat.Format32bppRgb))
        {
            var rect = new System.Drawing.Rectangle(0, 0, 640, 480);
            var data = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.WriteOnly, bmp.PixelFormat);
            System.Runtime.InteropServices.Marshal.Copy(c, 0, data.Scan0, c.Length);
            bmp.UnlockBits(data);
            bmp.Save(Path.Combine(folder, "color.png"), System.Drawing.Imaging.ImageFormat.Png);
        }
        var raw = new byte[d.Length * 2];
        Buffer.BlockCopy(d, 0, raw, 0, raw.Length);
        File.WriteAllBytes(Path.Combine(folder, "depth.raw"), raw);
        return 0;
    }

    static void OnColor(object s, ColorImageFrameReadyEventArgs e)
    {
        using (ColorImageFrame f = e.OpenColorImageFrame())
        {
            if (f == null) return;
            var buf = new byte[f.PixelDataLength];
            f.CopyPixelDataTo(buf);
            lock (Gate) { color = buf; colorStamp = DateTime.UtcNow.Ticks; }
        }
    }

    static void OnDepth(object s, DepthImageFrameReadyEventArgs e)
    {
        using (DepthImageFrame f = e.OpenDepthImageFrame())
        {
            if (f == null) return;
            var px = new DepthImagePixel[f.PixelDataLength];
            f.CopyDepthImagePixelDataTo(px);
            var mm = new short[px.Length];
            for (int i = 0; i < px.Length; i++) mm[i] = px[i].IsKnownDepth ? px[i].Depth : (short)0;
            lock (Gate) { depth = mm; depthStamp = DateTime.UtcNow.Ticks; }
        }
    }

    static void Serve(TcpClient client)
    {
        // A client that disconnects mid-request must only end its own thread, never the whole bridge.
        try { ServeClient(client); }
        catch (IOException) { }
        catch (SocketException) { }
        catch (ObjectDisposedException) { }
    }

    static void ServeClient(TcpClient client)
    {
        using (client)
        using (NetworkStream net = client.GetStream())
        using (var reader = new StreamReader(net, Encoding.ASCII))
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Trim().Split(' ');
                string cmd = parts[0].ToUpperInvariant();
                try
                {
                    if (cmd == "COLOR") SendColor(net);
                    else if (cmd == "DEPTH") SendDepth(net);
                    else if (cmd == "TILT") Reply(net, Tilt(int.Parse(parts[1])));
                    else if (cmd == "INFO") Reply(net, Info());
                    else if (cmd == "AUDIO") { StreamAudio(net); return; }
                    else Reply(net, "{\"err\":\"unknown\"}");
                }
                catch (Exception ex)
                {
                    Reply(net, "{\"err\":\"" + ex.Message.Replace("\"", "'") + "\"}");
                }
            }
        }
    }

    static void SendColor(NetworkStream net)
    {
        byte[] buf; long stamp;
        lock (Gate) { buf = color; stamp = colorStamp; }
        if (buf == null) { Reply(net, "{\"err\":\"no colour frame yet\"}"); return; }
        Reply(net, "{\"w\":640,\"h\":480,\"fmt\":\"bgra\",\"bytes\":" + buf.Length + ",\"ticks\":" + stamp + "}");
        net.Write(buf, 0, buf.Length);
    }

    static void SendDepth(NetworkStream net)
    {
        short[] mm; long stamp;
        lock (Gate) { mm = depth; stamp = depthStamp; }
        if (mm == null) { Reply(net, "{\"err\":\"no depth frame yet\"}"); return; }
        var buf = new byte[mm.Length * 2];
        Buffer.BlockCopy(mm, 0, buf, 0, buf.Length);
        Reply(net, "{\"w\":640,\"h\":480,\"fmt\":\"u16mm\",\"bytes\":" + buf.Length + ",\"ticks\":" + stamp + "}");
        net.Write(buf, 0, buf.Length);
    }

    static readonly object AudioGate = new object();

    // Windows' plain capture path gives this array almost no level; the SDK's audio source has gain control.
    static void StreamAudio(NetworkStream net)
    {
        if (!Monitor.TryEnter(AudioGate))
        {
            Reply(net, "{\"err\":\"audio already in use\"}");
            return;
        }
        KinectAudioSource src = sensor.AudioSource;
        try
        {
            src.EchoCancellationMode = EchoCancellationMode.None;
            src.AutomaticGainControlEnabled = true;
            src.NoiseSuppression = true;
            src.BeamAngleMode = BeamAngleMode.Adaptive;
            Stream audio = src.Start();
            Reply(net, "{\"rate\":16000,\"fmt\":\"s16le\",\"ch\":1}");
            var buf = new byte[3200];  // 100 ms
            while (true)
            {
                int n = audio.Read(buf, 0, buf.Length);
                if (n > 0) net.Write(buf, 0, n);
            }
        }
        finally
        {
            src.Stop();
            Monitor.Exit(AudioGate);
        }
    }

    static string Tilt(int degrees)
    {
        int clamped = Math.Max(sensor.MinElevationAngle, Math.Min(sensor.MaxElevationAngle, degrees));
        sensor.ElevationAngle = clamped;  // the motor is slow and wears out: change it rarely
        return "{\"ok\":\"tilt\",\"deg\":" + clamped + "}";
    }

    static string Info()
    {
        return "{\"status\":\"" + sensor.Status + "\",\"tilt\":" + sensor.ElevationAngle + "}";
    }

    static void Reply(NetworkStream net, string json)
    {
        byte[] b = Encoding.ASCII.GetBytes(json + "\n");
        net.Write(b, 0, b.Length);
    }
}
