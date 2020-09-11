Requirements:

keras
tensorflow
gym_super_mario_bros (gym + nes-py)
Windows Kits

In order to install nes-py on Windows:

0 - Install SDK Windows from: https://developer.microsoft.com/fr-fr/windows/downloads/windows-10-sdk/

1 - Locate rc.exe and rcdll.dll (on Windows Kits)

2 - Copy them to "Microsoft Visual Studio 14.0\VC\bin"


To fix the issue, do next steps:

    Add this to your PATH environment variables:

    C:\Program Files (x86)\Windows Kits\10\bin\x64

    Copy these files rc.exe & rcdll.dll from C:\Program Files (x86)\Windows Kits\8.1\bin\x86 to C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin

In newer Windows these files might also be in the highest version: C:\Program Files (x86)\Windows Kits\10\bin\10.0.VERSION\x86
