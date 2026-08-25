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
#ifndef MyAppVersion
  #define MyAppVersion   "1.0.0"
#endif
#ifndef MyAppVariant
  #define MyAppVariant   "cpu"
#endif
#if MyAppVariant == "vulkan"
  #define MyAppVariantSuffix "-vulkan"
#elif MyAppVariant == "cuda"
  #define MyAppVariantSuffix "-cuda"
#else
  #define MyAppVariantSuffix ""
#endif
#define MyAppPublisher "dagaza"
#define MyAppURL       "https://github.com/dagaza/Qube"
#define MyAppExeName   "Qube.exe"
; Keep in sync with core/windows_install_mutex.py INSTALL_MUTEX_NAME
#define MyAppMutex     "dagaza.Qube.AppMutex"
; One mutex for all CPU/Vulkan/CUDA installers (same AppId / install dir).
#define MySetupMutex   "dagaza.Qube.SetupMutex"

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
OutputBaseFilename=Qube-{#MyAppVersion}{#MyAppVariantSuffix}-Setup
OutputDir=..\installer\output
Compression=lzma2/ultra64
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
AppMutex={#MyAppMutex}
SetupMutex={#MySetupMutex}
#ifexist "..\assets\logos\qube.ico"
SetupIconFile=..\assets\logos\qube.ico
UninstallDisplayIcon={app}\Qube.exe
#endif

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

[Code]
const
  UninstallRegKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#SetupSetting("AppId")}_is1';

procedure KillRunningQube();
var
  ResultCode: Integer;
begin
  { /T terminates child processes so PyInstaller DLLs in _internal release. }
  Exec('taskkill.exe', '/F /IM {#MyAppExeName} /T', '', SW_HIDE,
    ewWaitUntilTerminated, ResultCode);
  Sleep(1000);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    KillRunningQube();
end;

function InitializeSetup(): Boolean;
var
  InstalledVersion: String;
begin
  Result := True;
  if RegQueryStringValue(HKCU, UninstallRegKey, 'DisplayVersion', InstalledVersion) then
  begin
    WizardForm.WelcomeLabel2.Caption :=
      'Setup will update Qube from version ' + InstalledVersion +
      ' to {#MyAppVersion}.' + #13#10 + #13#10 +
      'Your models, Library, memory, and settings in %LOCALAPPDATA%\Qube are kept.';
  end;
end;
