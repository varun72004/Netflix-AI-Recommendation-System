$ErrorActionPreference = "Stop"

function Test-PortAvailable {
    param([int]$Port)

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

$port = if (Test-PortAvailable 3000) { 3000 } else { 3001 }
$env:PORT = [string]$port

Write-Host "Starting CineAI at http://localhost:$port"
node server.js
