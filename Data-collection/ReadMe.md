# BehaVR Data Collection

In BehaVR, we instrumented ALVR to intercept and record sensor data in the SteamVR + Meta Quest Pro setup (see Section 3.1 in the paper). We provide relevant code in this folder.

## Hardware Setup

We set up the BehaVR data collection system with the following hardware:

- [Meta Quest Pro](https://www.meta.com/quest/quest-pro/) VR headset.
- Alienware x14 Gaming Laptop (Windows 11)
  - For running SteamVR and ALVR server. A high-end gaming PC is recommended.
- Macbook Pro 15-inch 2019 (macOS 12.1.1)
  - For data collection only. Any machine that can run `adb` should suffice.

The Quest Pro headset is connected to the gaming laptop through a 5GHz WiFi connection, which is the same as the usual ALVR setup. The headset is also connected to the MacBook through USB to enable `adb` connection.

## Modified ALVR Client

We modified the ALVR client app to intercept VR sensor data. We modified [ALXR](https://github.com/korejan/ALVR/tree/facial-eye-tracking), a forked version of ALVR that supports facial and eye tracking. The modification is provided as a patch file `openxr_program.patch`. After applying the patch to the ALXR code, one may follow [its documentation](https://github.com/korejan/ALVR/wiki/ALXR-Client#build-from-soure) to compile the APK package that will be installed on Quest Pro.

We tested the compilation on a Linux system. To facilitate replication, we provide an all-in-one script `build_apk.sh` that compiles the modified ALVR in a Docker (or Podman) container. We expect this to work on all platforms that support Docker. You may [install Docker](https://docs.docker.com/engine/install/) and compile the APK as follows:

```
$ git submodule init
$ git submodule update --recursive
$ cd Data-collection
$ bash build_apk.sh
```

If it runs successfully, you will find the generated APK `alxr-client-quest.apk` in the same folder.

## Setup ALVR Server and Client


Here, we set up ALVR pretty much in the same way as [the upstream](https://github.com/korejan/ALVR/wiki/Installation).

For the ALVR server, we use version 18.2.3 which can be downloaded [here](https://github.com/alvr-org/ALVR/releases/tag/v18.2.3).

For the ALVR client, we use the modified version that we compiled in the previous step. Connect the Quest Pro headset to a PC and install the APK file:

```
$ adb install -r alxr-client-quest.apk
```

## Data Collection

Once you start SteamVR, the ALVR server, and the (modified) ALVR client, you may use the following scripts to collect VR data.

`collect_data.sh` uses `adb logcat` to capture sensor readings and saves them into a file. Use it as follows:

```
$ collect_data.sh -f capture
```

`generate_all_csvs.sh` generates individual CSV files that record each type of sensor data:

```
$ generate_all_csvs.sh -f capture
```

This will generate a folder `capture/` containing the output CSV files.

## Notes on Reusability

The upstream ALVR/ALXR projects and SteamVR are all being continously developed. It is possible that the current version of ALVR that we relied on will become out-of-date and unusable in the future. In that case, we recommend you to manually port the few changes indicated in `openxr_program.patch` (specifically, code marked with `// TODO: Tap onto sensor value`) to the latest version of ALVR.
