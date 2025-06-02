let sessaoAtiva = false;
let slimSelectInstance = null;
let inatividadeTimer = null;

// Validar chave de acesso
function validarChave() {
  const senha = document.getElementById('chave').value.trim();
  const msg = document.getElementById('mensagem-chave');
  const spinner = document.getElementById('spinner');

  if (!senha) {
    msg.textContent = "❌ Digite a chave.";
    msg.style.color = "red";
    return;
  }

  spinner.style.display = "inline-block";

  const formData = new FormData();
  formData.append("senha", senha);

  fetch("https://learningbyworking.app.n8n.cloud/webhook/senha", {
    method: "POST",
    body: formData
  })
  .then(resposta => resposta.json())
  .then(raw => {
    const data = Array.isArray(raw) ? raw : [raw];
    const itemComNome = data.find(item => item?.nome?.trim());
    if (itemComNome) {
      msg.textContent = `✅ Acesso aprovado! Bem-vindo, ${itemComNome.nome}!`;
      msg.style.color = "green";
      // Desbloqueia campos
      ['prompt', 'arquivo', 'enviar', 'remover', 'ferramenta', 'grafico_tipo', 'coluna_y', 'colunas_x'].forEach(id => {
        document.getElementById(id).disabled = false;
      });
      document.getElementById('sair').style.display = 'inline-block';
      document.getElementById('chave').value = '';
      document.getElementById('chave').required = false;
      document.getElementById('chave').disabled = true;

      // Ativa multiselect em colunas_x se SlimSelect estiver presente
      if (typeof SlimSelect !== "undefined") {
        if (slimSelectInstance) slimSelectInstance.destroy();
        slimSelectInstance = new SlimSelect({
          select: '#colunas_x',
          settings: { closeOnSelect: false }
        });
      }
      sessaoAtiva = true;
      iniciarContadorInatividade();
    } else {
      msg.textContent = "❌ Chave incorreta ou sem autorização.";
      msg.style.color = "red";
    }
  })
  .catch(() => {
    msg.textContent = "❌ Erro na verificação.";
    msg.style.color = "red";
  })
  .finally(() => {
    spinner.style.display = "none";
  });
}

// Função para deslogar usuário
function deslogar() {
  if (!confirm("Tem certeza que deseja sair?\nTudo será apagado.")) return;
  sessaoAtiva = false;
  clearTimeout(inatividadeTimer);

  ['prompt', 'arquivo', 'enviar', 'remover', 'ferramenta', 'grafico_tipo', 'coluna_y', 'colunas_x'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.disabled = true;
      if (el.tagName === "SELECT" || el.tagName === "INPUT") el.value = '';
    }
  });

  if (slimSelectInstance) {
    slimSelectInstance.destroy();
    slimSelectInstance = null;
  }

  document.getElementById('sair').style.display = 'none';
  document.getElementById('mensagem-chave').textContent = '👋 Até a próxima!';
  document.getElementById('mensagem-chave').style.color = 'gray';
  document.getElementById('chave').required = true;
  document.getElementById('chave').disabled = false;
  document.getElementById('chave').value = '';
}

// Inatividade: deslogar após 10 min sem uso
function iniciarContadorInatividade() {
  clearTimeout(inatividadeTimer);
  inatividadeTimer = setTimeout(deslogar, 10 * 60 * 1000);
}
document.body.addEventListener('mousemove', iniciarContadorInatividade);
document.body.addEventListener('keydown', iniciarContadorInatividade);









