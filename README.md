# Python Virtual Environment Setup

This project uses a Python virtual environment called `.venv`.

## 1. Open the project folder

```powershell
cd S:\hologram-project
```

## 2. Create `.venv`

Run this only once:

```powershell
python -m venv .venv
```

## 3. Activate `.venv`

```powershell
.venv\Scripts\activate
```

After activation, you should see:

```text
(.venv) PS S:\hologram-project>
```

## 4. Install the required packages

```powershell
pip install opencv-python mediapipe numpy pygame scipy open3d
```

## 5. Run the project

For example:

```powershell
python hologram_3d.py
```

## 6. Deactivate `.venv`

When finished:

```powershell
deactivate
```

## Next time

You don't need to create `.venv` again.

Just run:

```powershell
cd S:\hologram-project
.venv\Scripts\activate
```

Then run your Python file:

```powershell
python hologram_3d.py
```

## Important

The `.venv` folder contains the project's Python environment and should generally **not be uploaded to GitHub**.

Add this to `.gitignore`:

```text
.venv/
```
