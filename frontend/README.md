# Scholar's Digital Binder frontend

Run commands from the project root (one directory above this file).

```powershell
python -m pip install -r requirements.txt
npm.cmd --prefix frontend install
npm.cmd --prefix frontend run build
python server.py
```

Open http://127.0.0.1:8000. FastAPI serves the built dashboard and its API together.
Configure API keys in the root `.env` file. Rebuild after editing frontend files.

For automatic refresh, keep the backend running and use a second terminal:

```powershell
npm.cmd --prefix frontend run dev
```

Open http://localhost:5173. Vite forwards API and figure requests to port 8000.
Stop each server with Ctrl+C.

## Original template notes

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and Oxlint's TypeScript related rules in your project.
