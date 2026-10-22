# Usuwa zadanie z Harmonogramu zadań Windows.
#   powershell -ExecutionPolicy Bypass -File scripts\usun_harmonogram.ps1
param([string]$TaskName = "Hotel overbooking - raport dzienny")
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "Usunięto zadanie '$TaskName'."
