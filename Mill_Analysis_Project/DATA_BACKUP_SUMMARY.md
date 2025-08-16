# DATA FILES BACKUP SUMMARY

## What was moved:
- **Reference file**: `Mill body - vale verde - v1 (1).PLY` (~8.8 MB)
- **Inner scan 1**: `8_8-Aug-2024.ply` (~1.0 GB)  
- **Inner scan 2**: `3_18-Jan-2024.ply` (~995 MB)
- **Total size**: ~2.0 GB

## Moved to:
`C:\Mill_Analysis_Project_DATA_BACKUP\`

## Project folder now:
- Size reduced by ~2GB 
- Ready for sharing/uploading to drive
- Contains all source code and configuration
- Placeholder files indicate where data went

## To restore functionality:
### Option 1: Copy files back
```bash
# Copy all files back to original locations
xcopy "C:\Mill_Analysis_Project_DATA_BACKUP\*" "C:\Mill_Analysis_Project\data\" /E /H /R /Y
```

### Option 2: Update config to point to backup
Edit `src/config.py` and change the file paths to point to:
- Reference: `C:\Mill_Analysis_Project_DATA_BACKUP\reference\Mill body - vale verde - v1 (1).PLY`
- Inner scans: `C:\Mill_Analysis_Project_DATA_BACKUP\inner_scans\`

## Backup folder structure:
```
Mill_Analysis_Project_DATA_BACKUP/
├── reference/
│   └── Mill body - vale verde - v1 (1).PLY
└── inner_scans/
    ├── 8_8-Aug-2024.ply
    └── 3_18-Jan-2024.ply
```

The project folder is now lightweight and ready for sharing!
