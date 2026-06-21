/**
 * primer_login.js
 * Detecta si el usuario entró por primera vez (flag en sessionStorage)
 * y muestra un modal para que cambie su contraseña por una personal.
 *
 * Uso: incluir este script en las páginas de alumno y docente DESPUÉS de config.js.
 * Requiere que el usuario esté en sessionStorage/localStorage como 'eco_user'
 * con la propiedad primer_login: true (seteada por el backend en /api/login).
 */

(function () {
    const STORAGE_KEY = 'eco_user';

    function getUser() {
        try {
            return JSON.parse(
                sessionStorage.getItem(STORAGE_KEY) ||
                localStorage.getItem(STORAGE_KEY) ||
                '{}'
            );
        } catch { return {}; }
    }

    function saveUser(user) {
        const dest = sessionStorage.getItem(STORAGE_KEY) ? sessionStorage : localStorage;
        dest.setItem(STORAGE_KEY, JSON.stringify(user));
    }

    function injectModal() {
        const html = `
<div id="primer-login-overlay" style="
    position:fixed;inset:0;background:rgba(0,0,0,0.65);
    display:flex;align-items:center;justify-content:center;
    z-index:9999;font-family:inherit;">
  <div style="
      background:#fff;border-radius:14px;padding:2rem 2.2rem;
      width:min(420px,90vw);box-shadow:0 8px 40px rgba(0,0,0,0.25);">
    <div style="font-size:1.4rem;font-weight:700;color:#1b5e20;margin-bottom:0.3rem;">
      🔐 Bienvenido/a
    </div>
    <p style="color:#555;font-size:0.9rem;margin-bottom:1.2rem;">
      Es tu <strong>primer acceso</strong>. Por seguridad, debes cambiar tu contraseña
      por una personal antes de continuar.
    </p>
    <div style="margin-bottom:0.9rem;">
      <label style="font-size:0.82rem;font-weight:600;color:#333;display:block;margin-bottom:4px;">Nueva contraseña</label>
      <input id="pl-nueva" type="password" placeholder="Nueva contraseña"
        style="width:100%;padding:0.55rem 0.75rem;border:1.5px solid #ccc;border-radius:8px;font-size:0.95rem;box-sizing:border-box;">
    </div>
    <div style="margin-bottom:1rem;">
      <label style="font-size:0.82rem;font-weight:600;color:#333;display:block;margin-bottom:4px;">Confirmar contraseña</label>
      <input id="pl-confirmar" type="password" placeholder="Repite la contraseña"
        style="width:100%;padding:0.55rem 0.75rem;border:1.5px solid #ccc;border-radius:8px;font-size:0.95rem;box-sizing:border-box;">
    </div>
    <div id="pl-error" style="display:none;padding:0.45rem 0.75rem;background:#ffebee;color:#c62828;border-radius:6px;font-size:0.82rem;margin-bottom:0.8rem;"></div>
    <button id="pl-btn" style="
        width:100%;padding:0.65rem;background:#2e7d32;color:#fff;
        border:none;border-radius:8px;font-size:1rem;font-weight:600;
        cursor:pointer;">Guardar contraseña</button>
  </div>
</div>`;
        document.body.insertAdjacentHTML('beforeend', html);
        document.getElementById('pl-btn').addEventListener('click', handleCambio);
    }

    async function handleCambio() {
        const nueva     = document.getElementById('pl-nueva').value;
        const confirmar = document.getElementById('pl-confirmar').value;
        const errorBox  = document.getElementById('pl-error');

        errorBox.style.display = 'none';

        if (nueva.length < 6) {
            errorBox.textContent = 'La contraseña debe tener al menos 6 caracteres.';
            errorBox.style.display = 'block';
            return;
        }
        if (nueva !== confirmar) {
            errorBox.textContent = 'Las contraseñas no coinciden.';
            errorBox.style.display = 'block';
            return;
        }

        const user = getUser();
        const rol  = user.rol;
        const id   = user.usuario_id;

        // Elegir endpoint según el rol
        const endpoint = rol === 'alumno'
            ? `${CONFIG.API_BASE_URL}/api/admin/alumnos/${id}`
            : `${CONFIG.API_BASE_URL}/api/admin/docentes/${id}`;

        const formData = new FormData();
        formData.append('contrasena_hash', nueva);

        // Para PUT necesitamos también nombre, apellido, aula_id (requeridos por el endpoint)
        // Los tomamos del mismo objeto de sesión
        formData.append('nombre',   user.nombre   || '');
        formData.append('apellido', user.apellido || '');
        formData.append('aula_id',  user.aula_id  || 0);

        try {
            const res = await fetch(endpoint, { method: 'PUT', body: formData });
            if (!res.ok) {
                const err = await res.json();
                errorBox.textContent = 'Error: ' + (err.detail || 'No se pudo guardar.');
                errorBox.style.display = 'block';
                return;
            }
            // Marcar primer_login como false en sesión
            user.primer_login = false;
            saveUser(user);
            document.getElementById('primer-login-overlay').remove();
        } catch (err) {
            errorBox.textContent = 'Error de red: ' + err.message;
            errorBox.style.display = 'block';
        }
    }

    // Esperar al DOM
    document.addEventListener('DOMContentLoaded', function () {
        const user = getUser();
        if (user.primer_login === true) {
            injectModal();
        }
    });
})();
