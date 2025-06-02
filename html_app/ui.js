// ❌ Modal de erro (popup)
function mostrarModalErro(msg) {
  let modal = document.getElementById("modal-erro");
  let conteudo = document.getElementById("modal-erro-texto");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "modal-erro";
    modal.style.position = "fixed";
    modal.style.top = "0";
    modal.style.left = "0";
    modal.style.width = "100%";
    modal.style.height = "100%";
    modal.style.background = "rgba(0,0,0,0.5)";
    modal.style.zIndex = "999";
    modal.style.display = "flex";
    modal.style.alignItems = "center";
    modal.style.justifyContent = "center";
    conteudo = document.createElement("div");
    conteudo.id = "modal-erro-texto";
    conteudo.style.background = "white";
    conteudo.style.padding = "20px 30px";
    conteudo.style.borderRadius = "8px";
    conteudo.style.textAlign = "center";
    conteudo.style.maxWidth = "400px";
    modal.appendChild(conteudo);
    document.body.appendChild(modal);
  }
  conteudo.textContent = msg;
  modal.style.display = "flex";
}

function fecharModalErro() {
  const modal = document.getElementById("modal-erro");
  if (modal) modal.style.display = "none";
}

