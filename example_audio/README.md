# Optional Example Audio

This folder is for local example-song audio only.

The public repository does not include commercial song MP3 files. If you have
licensed audio files, put them here with the example song ID as the filename:

```text
athiscode.mp3
brushupbrassup.mp3
divinespell.mp3
dokidokisingout.mp3
futarikoto.mp3
itsuaietara.mp3
kokokarakokokara.mp3
lemonsour.mp3
shunkansummerday.mp3
soundscape.mp3
```

Supported extensions:

```text
.mp3
.wav
.flac
.m4a
.ogg
```

After adding files, restart YesTiger and load an example in Web Studio. The
analysis timeline will load from the bundled example JSON, and the audio player
will use the matching local file.

You can also point YesTiger to another folder:

```powershell
$env:YESTIGER_EXAMPLE_AUDIO_DIR = "D:\path\to\example_audio"
.\start_windows.ps1
```
