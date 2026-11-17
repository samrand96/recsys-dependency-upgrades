$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$resultRoot = Join-Path $root 'results/issue-guided/L11'
New-Item -ItemType Directory -Path $resultRoot -Force | Out-Null
$result = Join-Path $resultRoot ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ'))
New-Item -ItemType Directory -Path $result | Out-Null
$commit = '5c862afa751f53559b5ad98b92e098760f530ad8'
$relativeSource = 'recommenders/models/tfidf/tfidf_utils.py'
$sourceUrl = "https://raw.githubusercontent.com/recommenders-team/recommenders/$commit/$relativeSource"
$blob = '6a6d22389714f13261dcb3751947617548e39cf4'
$source = Join-Path $result 'tfidf_utils_1.2.1.py'
Invoke-WebRequest -UseBasicParsing -Uri $sourceUrl -OutFile $source
if ((git hash-object --no-filters -- $source) -ne $blob) {
    throw 'Downloaded source does not match the tagged Git blob.'
}

$reproducer = Join-Path $PSScriptRoot 'recommenders_tfidf_self_item.py'
$image = docker image inspect silentrank-runner:py312 --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the Linux runner.' }
$oldId = 'py312-np1.26.4-sp1.13.1-pd2.2.3'
$newId = 'py312-np2.0.2-sp1.13.1-pd2.2.3'
$containerScript = @'
failures=0
for phase in old new rollback reapply
do
    case "$phase" in
        old|rollback) env_id=py312-np1.26.4-sp1.13.1-pd2.2.3 ;;
        new|reapply) env_id=py312-np2.0.2-sp1.13.1-pd2.2.3 ;;
    esac
    "/venvs/$env_id/bin/python" /repro.py /experiment/tfidf_utils_1.2.1.py "$phase" >"/experiment/$phase.json" 2>"/experiment/$phase.stderr.log"
    status=$?
    printf '%s\n' "$status" >"/experiment/$phase.exit"
    if [ "$status" -ne 0 ]; then failures=1; fi
done
exit "$failures"
'@
$shellScript = Join-Path $result 'run.sh'
$containerScript.Replace("`r`n", "`n") | Set-Content -LiteralPath $shellScript -Encoding utf8NoBOM -NoNewline
$dockerArgs = @(
    'run', '--rm', '--network', 'none', '-v', 'silentrank-venvs:/venvs:ro',
    '-v', "${result}:/experiment", '-v', "${reproducer}:/repro.py:ro", $image,
    '/bin/sh', '/experiment/run.sh'
)
$metadata = [ordered]@{
    planned_utc = [DateTime]::UtcNow.ToString('o')
    source_tag = '1.2.1'
    source_commit = $commit
    source_url = $sourceUrl
    source_git_blob = $blob
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    script_sha256 = (Get-FileHash -LiteralPath $reproducer -Algorithm SHA256).Hash.ToLowerInvariant()
    command_script_sha256 = (Get-FileHash -LiteralPath $shellScript -Algorithm SHA256).Hash.ToLowerInvariant()
    old_lock_sha256 = (Get-FileHash -LiteralPath (Join-Path $root "envs/$oldId/uv.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
    new_lock_sha256 = (Get-FileHash -LiteralPath (Join-Path $root "envs/$newId/uv.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
    image = $image
    scope = 'Tagged rank-selection stage with a frozen synthetic cosine matrix. Recommenders package installation and end-to-end TF-IDF are outside the treatment.'
    upstream_testsuite = 'not_runnable'
    upstream_testsuite_note = 'The existing N1 locks do not install the full Recommenders package or its test dependencies.'
    contract = 'All four queries must exclude the query item from their own top two.'
    fixture = 'Upstream PR 2391 A/B duplicate text, C/D disjoint, exact float64 cosine score matrix fixed in reproducer.'
    docker_argv = @('docker') + $dockerArgs
}
$metadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $result 'plan.json') -Encoding utf8NoBOM
Write-Output $result
docker @dockerArgs
if ($LASTEXITCODE -ne 0) { throw "A replay failed. See logs in $result." }
