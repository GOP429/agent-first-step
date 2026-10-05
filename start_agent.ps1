$ErrorActionPreference = "Stop"

$keyPath = Join-Path -Path $env:USERPROFILE -ChildPath ".deepseek_api_key_2026-09-02.encrypted"
$pythonPath = Join-Path -Path $PSScriptRoot -ChildPath ".venv\Scripts\python.exe"
$agentPath = Join-Path -Path $PSScriptRoot -ChildPath "langchain_agent.py"

if (-not (Test-Path -LiteralPath $keyPath)) {
    throw "Encrypted DeepSeek API key file was not found."
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project virtual environment was not found."
}

$encryptedKeyText = (Get-Content -Raw -LiteralPath $keyPath).Trim()
$secureKey = ConvertTo-SecureString -String $encryptedKeyText
$credential = [System.Management.Automation.PSCredential]::new("unused", $secureKey)

try {
    $env:DEEPSEEK_API_KEY = $credential.GetNetworkCredential().Password

    Write-Host "[...] Verifying DeepSeek API connection."

    & $pythonPath -c "from langchain_deepseek import ChatDeepSeek; ChatDeepSeek(model='deepseek-chat', temperature=0, max_tokens=1).invoke('Reply only with OK.')"

    if ($LASTEXITCODE -ne 0) {
        throw "DeepSeek API connection failed."
    }

    Write-Host "[OK] DeepSeek API connection verified." -ForegroundColor Green

    & $pythonPath $agentPath
}
finally {
    Remove-Item Env:\DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
}