; =========================================================
; Qube Installer — Inno Setup Script
; =========================================================
;
; Build with:  iscc installer\qube.iss
; Requires PyInstaller output in dist\Qube\ first.
;
; Silent install (WinGet / CI):
;   Qube-1.0.0-Setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
;

#define MyAppName      "Qube"
#define MyAppVersion   "1.0.0"
#define MyAppPublisher "dagaza"
#define MyAppURL       "https://github.com/dagaza/Qube"
#define MyAppExeName   "Qube.exe"

[Setup]
; NOTE: generate a unique AppId for your own fork — do NOT reuse this GUID.
AppId={{B7E4A3F1-92C0-4D8B-A6E5-3F1C7D9B0E42}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputBaseFilename=Qube-{#MyAppVersion}-Setup
OutputDir=..\installer\output
Compression=lzma2/ultra64
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
; Uncomment once an .ico is available:
; SetupIconFile=..\assets\logos\qube.ico
; UninstallDisplayIcon={app}\Qube.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\dist\Qube\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}";      Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
