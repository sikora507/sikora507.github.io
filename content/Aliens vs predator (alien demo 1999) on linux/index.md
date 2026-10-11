---
title: "Aliens versus Predator: running the 1999 Alien Demo on Linux"
date: 2026-10-04
tags:
  - linux
  - gaming
description: "How I got the original Alien Demo running on CachyOS with Bottles, Proton-CachyOS, dgVoodoo2 2.79.3, and DXVK-Sarek."
---

Getting an old game to display its menu is one thing. Getting into the actual level can be a different problem entirely. That was my experience with the original **Aliens versus Predator — Alien Demo**, a 32-bit Windows demo from January 1999, version 1.2.

After several attempts on CachyOS, I ended up with a configuration that ran the menu, videos, and full 3D gameplay. The breakthrough was **dgVoodoo2 2.79.3**, with its 32-bit libraries next to the game executable and an explicit Wine DLL override.

These notes describe the setup I confirmed working on **October 4, 2026**. I used an existing Wine prefix; I have not tested a fresh installation from scratch.

## The configuration that worked

The rendering path was:

```text
Alien Demo (DirectDraw / Direct3D)
    → dgVoodoo2 2.79.3 (Direct3D 11)
    → DXVK-Sarek (Vulkan)
    → NVIDIA GeForce GTX 1080 Ti
```

The exact environment matters, especially when comparing results with newer versions:

| Component | Tested configuration |
| --- | --- |
| Operating system | CachyOS |
| Desktop session | Hyprland / uwsm, Wayland with XWayland available (`DISPLAY=:1`) |
| Graphics card | NVIDIA GeForce GTX 1080 Ti |
| NVIDIA driver | 580.178.04 |
| Bottles | 67.3, official Flatpak (`com.usebottles.bottles`) |
| Runner | Proton-CachyOS Latest, build `cachyos-11.0-20260703-slr` |
| DXVK | DXVK-Sarek v1.12.0, included with the runner |
| dgVoodoo2 | **2.79.3**, x86 libraries |
| Game | Original Alien Demo, version 1.2, January 1999 |

The runner's name contains “Latest”, but that does not identify a fixed build. This is a record of one working combination, rather than a guarantee that every newer runner or wrapper will behave the same way.

## Setting it up

### 1. Keep the existing game and Wine prefix together

In my installation, the game directory also contains the Wine prefix:

```text
/home/tomasz/Documents/Gry/AliensVersusPredator-AlienDemo
```

Alongside `AvP_Alien_Demo.exe` and the game assets, it contains `drive_c`, `system.reg`, `user.reg`, and Proton files. If your installation has the same layout, preserve the whole directory. Do not delete the prefix files or initialize a new prefix over them.

Before changing the setup, make a copy of the directory with the game closed. The steps below assume this existing installation; a clean prefix may need additional setup that I have not verified.

### 2. Select the runner in Bottles

My game entry in Bottles is named `Alien`. It uses this runner:

```text
/home/tomasz/.local/share/Steam/compatibilitytools.d/Proton-CachyOS Latest
```

The executable is:

```text
/home/tomasz/Documents/Gry/AliensVersusPredator-AlienDemo/AvP_Alien_Demo.exe
```

I left the arguments empty and did not set a separate working directory. The game sandbox was enabled, and network sharing was disabled. Adapt the paths to your own installation.

### 3. Get dgVoodoo2 2.79.3

Version **2.79.3** made the difference in my tests. Version 2.87.5 crashed in this setup even after I confirmed that Wine was loading its libraries.

I downloaded the older archive from the third-party `legluondunet/mlls-tools` repository:

```bash
curl -fL https://raw.githubusercontent.com/legluondunet/mlls-tools/master/dgVoodoo2/dgVoodoo2_79_3.zip \
  -o /tmp/dgVoodoo2_79_3.zip
```

The SHA-256 of the archive I used was:

```text
67e5ff5f647555f5cd39c6cee9ffa2ecadcd7c9f4fe3b75e33c17486223e2d41
```

This identifies my copy; it is not an independent authenticity check from the dgVoodoo author.

### 4. Copy the x86 libraries next to the executable

Extract these files into the directory containing `AvP_Alien_Demo.exe`:

| File in the archive | Destination in the game directory |
| --- | --- |
| `MS/x86/DDraw.dll` | `DDraw.dll` |
| `MS/x86/D3DImm.dll` | `D3DImm.dll` |
| `dgVoodooCpl.exe` | `dgVoodooCpl.exe` |

The demo is a 32-bit application, so use the **x86** DLLs. I did not copy `D3D8.dll` or `D3D9.dll`, and I did not manually replace any libraries in the prefix's system directory.

`dgVoodooCpl.exe` is the settings panel. Continue launching the game through `AvP_Alien_Demo.exe`.

### 5. Configure dgVoodoo

These are the relevant settings from my working `dgVoodoo.conf`, stored next to the executable:

```ini
Version = 0x279

[General]
OutputAPI = d3d11_fl11_0
FullScreenMode = false

[GeneralExt]
PresentationModel = discard

[DirectX]
VideoCard = geforce_ti_4800
VRAM = 256
Resolution = unforced
Antialiasing = off
AppControlledScreenMode = false
dgVoodooWatermark = true

[DirectXExt]
DefaultEnumeratedResolutions = classics
EnumeratedResolutionBitdepths = all
Default3DRenderFormat = argb8888
```

This is an **excerpt**, not the complete configuration file. I kept a full local backup for exact reproduction, listed below.

`geforce_ti_4800` is the card presented to the old game; the physical GPU doing the rendering is still the GTX 1080 Ti. The windowed settings were part of the successful test, although Wine may still show a virtual desktop instead of an ordinary application window. I did not change the Linux desktop's color depth.

### 6. Set the environment variables in Bottles

The successful run used these values:

| Variable | Value |
| --- | --- |
| `PROTON_DXVK_SAREK` | `1` |
| `WINEDLLOVERRIDES` | `ddraw,d3dimm=n` |
| `__GLX_VENDOR_LIBRARY_NAME` | `nvidia` |
| `PROTON_LOG` | `1` |
| `PROTON_LOG_DIR` | `/home/tomasz/Documents/Gry/AliensVersusPredator-AlienDemo` |
| `WINEDEBUG` | `+timestamp,+pid,+tid,+seh,+winediag,+loaddll` |

The crucial override is `WINEDLLOVERRIDES=ddraw,d3dimm=n`. Here, `n` selects native Windows DLLs: the dgVoodoo libraries copied beside the executable. Copying the files alone was insufficient in my tests; without the override, Wine continued to load its own DirectDraw implementation.

If you have been trying other rendering paths, remove these experimental variables before reproducing the setup:

- `PROTON_D7VK_DDRAW`
- `WINE_D3D_CONFIG`, including `renderer=gdi` or `renderer=vulkan`
- `WINE_USE_EGL`
- `PROTON_ENABLE_WAYLAND`
- `LIBGL_ALWAYS_SOFTWARE`

Save the settings before launching. A few of my earlier attempts did not test the intended configuration because a runner selection or variable had not been saved.

### 7. Test the actual level

Launch the `Alien` entry in Bottles and choose **Start Alien Demo**. Check the gameplay as well as the menu and videos: several earlier configurations could display those but failed when starting the level.

With dgVoodoo2 2.79.3 in place, all three worked.

## What failed along the way

The first failure appeared after Flatpak and driver updates: the game exited with code 1, and the log reported `GLXMakeCurrent: BadAlloc` in Wine's OpenGL path. The timing suggested a regression, but I did not establish which update caused it.

Tests of 32-bit OpenGL on both the host and inside Flatpak passed on NVIDIA. That made a general absence of 32-bit OpenGL support an unlikely explanation.

| Attempt | Result |
| --- | --- |
| Runtime and sandbox changes, software rendering, EGL, native Wayland | No working gameplay; some attempts produced an empty virtual desktop or only a working menu. |
| Proton-CachyOS with Sarek and D7VK | Did not solve the complete problem. |
| `WINE_D3D_CONFIG=renderer=gdi` | Menu and videos worked, but starting the level reported missing hardware DirectDraw support. |
| `WINE_D3D_CONFIG=renderer=vulkan` | Failed while creating a swapchain for the 16-bit `B5G6R5` format. Switching to a 32-bit mode did not fix it. |
| dgVoodoo2 2.87.5 without DLL overrides | Wine did not use the copied dgVoodoo libraries. |
| dgVoodoo2 2.87.5 with DLL overrides | Loaded correctly and created a D3D11 device through DXVK, then crashed inside its `DDraw.dll`. |
| dgVoodoo2 2.79.3 with its configuration and DLL overrides | Menu, videos, and 3D gameplay worked. |

The newer dgVoodoo failure was the useful turning point: it showed that the wrapper was being loaded, but that this version still failed in my environment. Replacing the libraries and configuration with 2.79.3 resolved it.

I have not isolated every setting to prove which ones are essential. The configuration above preserves the successful test without further simplification. Logging and the dgVoodoo watermark were still enabled; disabling them would be a separate test.

## Local backups and useful logs

For my own future troubleshooting, I kept the following files. These are paths in my installation, not downloads bundled with this article.

| Path | Contents or purpose |
| --- | --- |
| `avp-dzialajaca-konfiguracja-2026-10-04/` | The 2.79.3 archive, full `dgVoodoo.conf`, and a Bottles `game.yml` settings snapshot. |
| `dgvoodoo-backup-2.87.5/` | The previous DLLs, control panel, and configuration. |
| `steam-default.log` | Proton log, overwritten on subsequent runs. |
| `/tmp/bottles-avp.log` | Bottles launch log; `/tmp` is not permanent storage. |
| `drive_c/users/runner/AppData/Local/Temp/dx_error.log` | The game's current error log. |

The `dx_error.log` in the top-level game directory is an old file from 1999, so it is easy to inspect the wrong log. The `game.yml` snapshot is documentation of the setup; paths and the game entry identifier need adapting before using it elsewhere.

## References

- [dgVoodoo2 repository](https://github.com/dege-diosg/dgVoodoo2)
- [dgVoodoo2 general documentation](https://dege.freeweb.hu/dgVoodoo2/ReadmeGeneral/)
- [DXVK issue discussing newer dgVoodoo versions under Wine](https://github.com/doitsujin/dxvk/issues/5217)
- [Third-party archive used for the 2.79.3 download](https://github.com/legluondunet/mlls-tools/tree/master/dgVoodoo2)

The reported problems with newer dgVoodoo versions gave me a reason to try an older release. Similar symptoms do not establish that my crash had the same underlying cause.
