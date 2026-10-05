# Python Virtual Environment Setup

This project uses a Python virtual environment called `.venv`.

## 1. Open the project folder

Open PowerShell and run:

```powershell
cd S:\hologram-project
```

## 2. Create `.venv`

Run:

```powershell
python -m venv .venv
```

This creates:

```text
hologram-project/
└── .venv/
```

## 3. Activate `.venv`

For PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see:

```text
(.venv) PS S:\hologram-project>
```

## 4. Install the required packages

With `.venv` activated:

```powershell
pip install opencv-python mediapipe numpy pygame scipy open3d
```

## 5. Deactivate `.venv`

When you're finished:

```powershell
deactivate
```

## Next time

You **do not need to create `.venv` again**.

Just open the project folder and activate it:

```powershell
cd S:\hologram-project
.\.venv\Scripts\Activate.ps1
```

Then run your Python files normally:

```powershell
python hologram_3d.py
```

## If PowerShell blocks activation

Run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```
