# Items to REMOVE before GitHub upload

## Large folders/files (GitHub file size limits):
- `mill_analysis_env/` - Virtual environment (~200MB-1GB)
- `output/` - Generated visualizations and results
- `backup_old_results/` - Contains PLY files (~1GB+)
- `intermediate/` - Temporary processing files

## Python cache files:
- `__pycache__/` folders
- `*.pyc` files
- `*.pyo` files

## OS/IDE files:
- `.DS_Store` (macOS)
- `Thumbs.db` (Windows)
- `.vscode/` (VS Code settings)
- `.idea/` (PyCharm settings)

## Log/temp files:
- `*.log`
- `*.tmp`
- `*.temp`

# Commands to clean up:

## Remove large directories:
```bash
rmdir mill_analysis_env /s /q
rmdir output /s /q
rmdir backup_old_results /s /q
rmdir intermediate /s /q
```

## Remove cache files:
```bash
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
del /s /q *.pyc
del /s /q *.pyo
```

After cleanup, your repo will be ~10-50MB instead of 1GB+
