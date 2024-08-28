#!/bin/bash

### Self-contained script to build the patched ALXR APK in a container

set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-x}" != "build" ]]; then
    CMD="$((command -v docker podman || true) | grep -m1 .)"

    "$CMD" run --rm -ti --name behavr \
        -v "$PWD":/data \
        -w /data ubuntu:22.04 /bin/bash "$(basename "$0")" build
    ls alxr-client-quest.apk
    exit
fi

### One may follow the commands below to compile the APK on Ubuntu 22.04

# System dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -q -y curl clang gcc-multilib sdkmanager git build-essential openjdk-11-jdk-headless

# Android SDK
sdkmanager --install "ndk;25.2.9519653" "platforms;android-32" "cmake;3.22.1" "build-tools;32.0.0"
(yes || true) | sdkmanager --licenses
export ANDROID_HOME=/opt/android-sdk/ ANDROID_NDK_ROOT=/opt/android-sdk/ndk/25.2.9519653/

# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build ALXR
cp -Ta ALXR /root/ALXR

cd /root/ALXR
patch -p1 < /data/openxr_program.patch
cargo xtask build-alxr-quest --release --no-nvidia --oculus-ext

# Copy the APK out
cp ./target/quest/release/apk/alxr-client-quest.apk /data
