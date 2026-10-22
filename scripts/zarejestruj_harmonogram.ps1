# Rejestruje codzienne zadanie w Harmonogramie zadań Windows (dla bieżącego użytkownika).
# Uruchomienie (z folderu projektu):
#   powershell -ExecutionPolicy Bypass -File scripts\zarejestruj_harmonogram.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\zarejestruj_harmonogram.ps1 -Time 07:30
param(
    [string]$Time = "08:00",
    [string]$TaskName = "Hotel overbooking - raport dzienny"
)
$root = Split-Path -Parent $PSScriptRoot
$bat = Join-Path $root "scripts\daily_run.bat"
if (-not (Test-Path (Join-Path $root ".venv\Scripts\python.exe"))) {
    Write-Warning "Nie znaleziono .venv w $root - utwórz środowisko: python -m venv .venv; pip install -r requirements.txt"
}
$action   = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`"" -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -Daily -At $Time
# StartWhenAvailable: jeśli komputer był wyłączony o 8:00, zadanie wykona się zaraz po włączeniu
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Plan overbookingu i lista rezerwacji do kontaktu (outputs\site\index.html)" -Force | Out-Null

Write-Host "Zarejestrowano zadanie '$TaskName' - codziennie o $Time."
Write-Host "Test teraz:   Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Log:          $root\logs\daily_run.log"
Write-Host "Raport:       $root\outputs\site\index.html"
