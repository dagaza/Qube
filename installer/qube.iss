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
CloseApplications=force
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

var
  DeleteUserData: Boolean;

procedure KillRunningQube();
var
  ResultCode: Integer;
  Attempt: Integer;
begin
  { /T terminates child processes so PyInstaller DLLs in _internal release. }
  for Attempt := 1 to 12 do
  begin
    Exec('taskkill.exe', '/F /IM {#MyAppExeName} /T', '', SW_HIDE,
      ewWaitUntilTerminated, ResultCode);
    if not CheckForMutexes('{#MyAppMutex}') then
      Break;
    Sleep(500);
  end;
  Sleep(1500);
end;

function InitializeUninstall(): Boolean;
begin
  { Must run before AppMutex check or file removal while Qube is in the tray. }
  KillRunningQube();
  DeleteUserData := False;
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  LocalData, DotQube: String;
begin
  if CurUninstallStep = usUninstall then
  begin
    KillRunningQube();
    if not UninstallSilent() then
      DeleteUserData := MsgBox(
        'Remove your Qube data as well?' + #13#10 + #13#10 +
        'This deletes models, library, memory, and settings under:' + #13#10 +
        '  %LOCALAPPDATA%\Qube' + #13#10 +
        '  %USERPROFILE%\.qube' + #13#10 + #13#10 +
        'Choose No to keep your data for a future reinstall.',
        mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
  end;
  if (CurUninstallStep = usPostUninstall) and DeleteUserData then
  begin
    LocalData := ExpandConstant('{localappdata}') + '\Qube';
    DotQube := ExpandConstant('{userprofile}') + '\.qube';
    if DirExists(LocalData) then
      DelTree(LocalData, True, True, True);
    if DirExists(DotQube) then
      DelTree(DotQube, True, True, True);
  end;
end;

#if MyAppVariant == "cuda"
procedure CurStepChanged(CurStep: TSetupStep);
var
  InstallMark: String;
begin
  if CurStep = ssPostInstall then
  begin
    InstallMark := ExpandConstant('{app}\.qube-install-ts');
    SaveStringToFile(InstallMark, '1', False);
  end;
end;
#endif

function InitializeSetup(): Boolean;
var
  InstalledVersion: String;
begin
  { Must run before AppMutex check. Silent/WinGet upgrade hangs forever if Qube.exe
    still holds dagaza.Qube.AppMutex (suppressed "please close Qube" dialog). }
  KillRunningQube();
  Result := True;
  if RegQueryStringValue(HKCU, UninstallRegKey, 'DisplayVersion', InstalledVersion) then
  begin
    WizardForm.WelcomeLabel2.Caption :=
      'Setup will update Qube from version ' + InstalledVersion +
      ' to {#MyAppVersion}.' + #13#10 + #13#10 +
      'Your models, Library, memory, and settings are kept under' + #13#10 +
      '%LOCALAPPDATA%\Qube and %USERPROFILE%\.qube.';
  end;
end;
