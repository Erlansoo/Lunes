"""The cameras Windows has, by name, so Lunes opens the microscope and the bench camera by what they are and not by
a number that changes when another camera is plugged in.

The list comes from DirectShow, in the same order OpenCV numbers them with cv2.CAP_DSHOW.

    python -m lunes.cameras
"""
import ctypes
from ctypes import POINTER, c_ulong, c_void_p

import comtypes
import comtypes.client
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
from comtypes.automation import VARIANT
from comtypes.persist import IPropertyBag

SYSTEM_DEVICE_ENUM = GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
VIDEO_INPUT_CATEGORY = GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")


class IMoniker(IUnknown):
    # only up to BindToStorage, the one used; the methods before it keep the table's order
    _iid_ = GUID("{0000000F-0000-0000-C000-000000000046}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetClassID", (["out"], POINTER(GUID), "pClassID")),
        COMMETHOD([], HRESULT, "IsDirty"),
        COMMETHOD([], HRESULT, "Load", (["in"], c_void_p, "pStm")),
        COMMETHOD([], HRESULT, "Save", (["in"], c_void_p, "pStm"), (["in"], ctypes.c_int, "fClearDirty")),
        COMMETHOD([], HRESULT, "GetSizeMax", (["out"], POINTER(ctypes.c_ulonglong), "pcbSize")),
        COMMETHOD([], HRESULT, "BindToObject", (["in"], c_void_p, "pbc"), (["in"], c_void_p, "pmkToLeft"),
                  (["in"], POINTER(GUID), "riidResult"), (["out"], POINTER(c_void_p), "ppvResult")),
        COMMETHOD([], HRESULT, "BindToStorage", (["in"], c_void_p, "pbc"), (["in"], c_void_p, "pmkToLeft"),
                  (["in"], POINTER(GUID), "riid"), (["out"], POINTER(POINTER(IPropertyBag)), "ppvObj")),
    ]


class IEnumMoniker(IUnknown):
    _iid_ = GUID("{00000102-0000-0000-C000-000000000046}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Next", (["in"], c_ulong, "celt"), (["out"], POINTER(POINTER(IMoniker)), "rgelt"),
                  (["out"], POINTER(c_ulong), "pceltFetched")),
    ]


class ICreateDevEnum(IUnknown):
    _iid_ = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
    _methods_ = [
        COMMETHOD([], HRESULT, "CreateClassEnumerator", (["in"], POINTER(GUID), "clsidDeviceClass"),
                  (["out"], POINTER(POINTER(IEnumMoniker)), "ppEnumMoniker"), (["in"], c_ulong, "dwFlags")),
    ]


def cameras():
    """[(name, USB path)] of every camera, in OpenCV's DirectShow order. The path holds the USB id
    (vid_1224&pid_2825), which tells apart two cameras with the same generic name."""
    comtypes.CoInitialize()
    devices = comtypes.client.CreateObject(SYSTEM_DEVICE_ENUM, interface=ICreateDevEnum)
    monikers = devices.CreateClassEnumerator(VIDEO_INPUT_CATEGORY, 0)
    found = []
    while monikers:  # None when there is no camera at all
        moniker, fetched = monikers.Next(1)
        if not fetched:
            break
        bag = moniker.BindToStorage(None, None, IPropertyBag._iid_)
        props = []
        for prop in ("FriendlyName", "DevicePath"):
            try:
                props.append(str(bag.Read(prop, VARIANT(), None)))
            except comtypes.COMError:  # virtual cameras (OBS) have no USB path
                props.append("")
        found.append(tuple(props))
    return found


def find_camera(wanted):
    """OpenCV's number for a camera given as its number or as part of its name or USB path; None if it is not
    plugged in (or, for the Sony, not in USB Streaming mode)."""
    if str(wanted).strip().isdigit():
        return int(wanted)
    wanted = str(wanted).lower()
    for index, (name, path) in enumerate(cameras()):
        if wanted in name.lower() or wanted in path.lower():
            return index
    return None


if __name__ == "__main__":
    for i, (name, path) in enumerate(cameras()):
        print(i, name, path)
