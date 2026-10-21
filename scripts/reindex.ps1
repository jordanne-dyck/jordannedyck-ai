<#
Rebuild the local FAISS index and restart the backend on it.

Run by hand, or by the post-commit hook in knowledge-base\ (scripts\kb-hooks).
A commit in knowledge-base\ is the approval: nothing here runs on its own.

  powershell -ExecutionPolicy Bypass -File C:\Users\Jord\jordannedyck-ai\scripts\reindex.ps1 -Label client-wording -Expect "counsellor who is starting"

Order matters. The running server read the index files at startup and holds
them in memory, so it keeps serving the old index while the embed runs. It is
only stopped once the new index exists and has passed the -Expect check.

This updates the LOCAL index only. The cluster PVC is re-seeded separately.
#>
param(
    [string]$Label = "reindex",
    [string[]]$Expect = @(),
    # Set by the hook. Reads "expect: <phrase>" lines from that commit's message.
    [string]$Commit = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root "venv\Scripts\python.exe"
$Db = Join-Path $Root "faiss_db"
$Archive = Join-Path $Root "knowledge-base-archive"
$Kb = Join-Path $Root "knowledge-base"
Set-Location $Root

if ($Commit) {
    $Label = "kb-$Commit"
    $msg = git -C $Kb log -1 --format=%B $Commit
    $Expect = @($msg | Where-Object { $_ -match '^\s*expect:\s*(.+?)\s*$' } | ForEach-Object { $Matches[1] })
}

function Fail($msg) {
    Write-Host "`n[reindex] FAILED: $msg" -ForegroundColor Red
    exit 1
}

function Get-ApiServer {
    # The process listening on 5000, only if it is this repo's api_server.py.
    $conn = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $conn) { return $null }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $($conn.OwningProcess)"
    [pscustomobject]@{ Pid = $conn.OwningProcess; Cmd = $proc.CommandLine }
}

function Get-Health {
    try { Invoke-RestMethod http://localhost:5000/health -TimeoutSec 5 } catch { $null }
}

# 1. Warn about edits that are on disk but not committed. The embed reads the
#    folder, not the commit, so these WILL go into the index.
if (Test-Path (Join-Path $Kb ".git")) {
    $dirty = git -C $Kb status --porcelain -- "*.md"
    if ($dirty) {
        Write-Host "[reindex] WARNING: uncommitted knowledge-base edits will be indexed too:" -ForegroundColor Yellow
        $dirty | ForEach-Object { Write-Host "    $_" }
    }
}

$before = Get-Health
$oldVersion = if ($before) { $before.index_version } else { "(server not running)" }
Write-Host "[reindex] running index_version: $oldVersion"

# 2. Back up the current index. The timestamp makes the name unique.
$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$backup = Join-Path $Archive "faiss_db_pre-${Label}_$stamp"
Copy-Item -Recurse $Db $backup
Write-Host "[reindex] backup: $backup"

# 3. Embed. Costs OpenAI embedding calls, one per chunk.
& $Python (Join-Path $Root "scripts\embed_knowledge_faiss.py")
if ($LASTEXITCODE -ne 0) { Fail "embed script exited $LASTEXITCODE. faiss_db\ may be partly written; the backup above is intact. The server was not touched." }

# 4. Check the new text is in the index, by reading the actual chunks.
$counts = @{}
foreach ($phrase in $Expect) {
    $n = & $Python -c "import pickle,sys; d=pickle.load(open(sys.argv[1],'rb')); print(sum(sys.argv[2] in c for c in d))" (Join-Path $Db "documents.pkl") $phrase
    $counts[$phrase] = [int]$n
    if ([int]$n -eq 0) { Fail "'$phrase' is in 0 chunks of the new index. The server was not restarted and still serves the old index. To restore: Remove-Item -Recurse $Db; Copy-Item -Recurse $backup $Db" }
}

# 5. Restart the backend on the new index, in its own window.
$srv = Get-ApiServer
if ($srv) {
    if ($srv.Cmd -notmatch "api_server\.py") { Fail "port 5000 is held by something that is not api_server.py (pid $($srv.Pid)): $($srv.Cmd). Not stopping it. The new index is built; restart the backend yourself." }
    Stop-Process -Id $srv.Pid -Force
    Write-Host "[reindex] stopped api_server.py (pid $($srv.Pid))"
    Start-Sleep -Seconds 2
}
Start-Process -FilePath $Python -ArgumentList "api_server.py" -WorkingDirectory $Root
Write-Host "[reindex] started api_server.py in a new window; waiting for /health"

$after = $null
for ($i = 0; $i -lt 30 -and -not $after; $i++) { Start-Sleep -Seconds 2; $after = Get-Health }
if (-not $after) { Fail "backend did not answer /health within 60 seconds. Look at the new server window for the error." }

# 6. Report actual values, not a pass/fail.
Write-Host "`n[reindex] done" -ForegroundColor Green
Write-Host "  index_version  $oldVersion -> $($after.index_version)"
Write-Host "  chunks         $($after.chunks)"
Write-Host "  health ok      $($after.ok)  (openai: $($after.openai))"
foreach ($phrase in $Expect) { Write-Host "  '$phrase' in $($counts[$phrase]) chunk(s)" }
if (-not $Expect) { Write-Host "  no -Expect phrase given, so the new text was not checked" -ForegroundColor Yellow }
Write-Host "  backup         $backup"
Write-Host "  LOCAL only. The cluster PVC is unchanged."
