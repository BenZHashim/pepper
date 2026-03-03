# test.ps1 — Simulate SMS messages locally
param(
    [string]$Message = "hello",
    [string]$From = "+13105559876"
)

$encodedFrom = $From -replace '\+', '%2B'
$encodedBody = [System.Uri]::EscapeDataString($Message)

Write-Host "Sending: $Message" -ForegroundColor Cyan

$response = Invoke-WebRequest `
  -Uri "http://localhost:5000/sms" `
  -Method POST `
  -ContentType "application/x-www-form-urlencoded" `
  -Body "From=$encodedFrom&Body=$encodedBody" `
  -UseBasicParsing

# Extract and print just the AI reply from the TwiML response
$xml = [xml]$response.Content
$reply = $xml.Response.Message
Write-Host "Reply: $reply" -ForegroundColor Green