$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$results = Join-Path $root 'results/issue-guided/L7'
New-Item -ItemType Directory -Path $results -Force | Out-Null
$result = Join-Path $results ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ'))
New-Item -ItemType Directory -Path $result | Out-Null

$commit = '8a95dbe24ca675a6edd86aafb3b4cd5ae7287edf'
$sourceUrl = "https://raw.githubusercontent.com/benfred/implicit/$commit/implicit/nearest_neighbours.py"
$sourceBlob = '29e50c2ce1496b3973387c69417ef07df5454736'
$source = Join-Path $result 'nearest_neighbours_0.7.3.py'
Invoke-WebRequest -UseBasicParsing -Uri $sourceUrl -OutFile $source
if ((git hash-object --no-filters -- $source) -ne $sourceBlob) {
    throw 'Downloaded implicit source does not match the tagged Git blob.'
}

$wheelAssets = @(
    @{url = 'https://files.pythonhosted.org/packages/0f/50/de23fde84e45f5c4fda2488c759b69990fd4512387a8632860f3ac9cd225/numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'; sha256 = '675d61ffbfa78604709862923189bad94014bef562cc35cf61d3a07bba02a7ed'},
    @{url = 'https://files.pythonhosted.org/packages/87/8c/97e545034c94d0bbbc3af3202551c3d6020e5f8d2ee37ebcabd9a2048174/scipy-1.13.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'; sha256 = '1e7626dfd91cdea5714f343ce1176b6c4745155d234f1033584154f60ef1ff42'}
)
foreach ($asset in $wheelAssets) {
    $filename = $asset.url.Split('/')[-1]
    $destination = Join-Path $result $filename
    Invoke-WebRequest -UseBasicParsing -Uri $asset.url -OutFile $destination
    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $asset.sha256) {
        throw "Downloaded wheel does not match published SHA-256 for $filename."
    }
}

$oldId = 'py312-np1.26.4-sp1.12.0-pd2.2.3'
$repairId = 'py312-np1.26.4-sp1.13.1-pd2.2.3'
$reproducer = Join-Path $PSScriptRoot 'scipy_sparse_duplicates.py'
$image = docker image inspect silentrank-runner:py312 --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the Linux runner.' }
$containerScript = @'
set -eu
uv venv /tmp/l7 > /experiment/build.log 2>&1
uv pip install --python /tmp/l7/bin/python --no-index --no-deps /experiment/*.whl >> /experiment/build.log 2>&1
uv pip freeze --python /tmp/l7/bin/python > /experiment/new-freeze.txt
failures=0
for phase in old new repair rollback
do
    case "$phase" in
        old|rollback) python=/venvs/py312-np1.26.4-sp1.12.0-pd2.2.3/bin/python ;;
        new) python=/tmp/l7/bin/python ;;
        repair) python=/venvs/py312-np1.26.4-sp1.13.1-pd2.2.3/bin/python ;;
    esac
    status=0
    "$python" /repro.py /experiment/nearest_neighbours_0.7.3.py "$phase" > "/experiment/$phase.json" 2> "/experiment/$phase.stderr.log" || status=$?
    printf '%s\n' "$status" > "/experiment/$phase.exit"
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
    source_tag = 'implicit v0.7.3'
    source_commit = $commit
    source_url = $sourceUrl
    source_git_blob = $sourceBlob
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    script_sha256 = (Get-FileHash -LiteralPath $reproducer -Algorithm SHA256).Hash.ToLowerInvariant()
    command_script_sha256 = (Get-FileHash -LiteralPath $shellScript -Algorithm SHA256).Hash.ToLowerInvariant()
    old_lock_sha256 = (Get-FileHash -LiteralPath (Join-Path $root "envs/$oldId/uv.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
    repair_lock_sha256 = (Get-FileHash -LiteralPath (Join-Path $root "envs/$repairId/uv.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
    new_wheels = $wheelAssets
    image = $image
    scope = 'Synthetic CSR construction and source-extracted implicit cosine normalization. RecTools, compiled neighbor selection, rankings, and relevance metrics are outside this component check.'
    upstream_testsuite = 'not_runnable'
    upstream_testsuite_note = 'This replay extracts the tagged implicit normalizer without installing the full package or its test dependencies.'
    contract = 'Both normalized nonzero item vectors have L2 norm 1 within absolute tolerance 1e-12.'
    fixture = 'User-item coordinates (0,0),(0,0),(1,0),(0,1),(1,1) in a 2x2 matrix, with float64 weight 1 at every input coordinate.'
    docker_argv = @('docker') + $dockerArgs
}
$metadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $result 'plan.json') -Encoding utf8NoBOM
Write-Output $result
docker @dockerArgs
if ($LASTEXITCODE -ne 0) { throw "An environment build or replay failed. See logs in $result." }
