let sessaoAtiva = false;
let inatividadeTimer = null;
let slimSelectInstance = null;

// ⏱️ Timer de inatividade
function resetarTimer() {
  if (!sessaoAtiva) return;
  clearTimeout(inatividadeTimer);
  inatividadeTimer = setTimeout(() => {
    deslogar();
    document.getElementById('mensagem-chave').textContent = "⏱ Sessão expirada por inatividade. Por favor, insira a chave novamente.";
    document.getElementById('mensagem-chave').style.color = "orange";
  }, 10 * 60 * 1000);
}

function iniciarMonitoramentoInatividade() {
  ['mousemove', 'keydown', 'scroll'].forEach(evt =>
    document.addEventListener(evt, resetarTimer)
  );
  document.addEventListener('click', e => {
    if (e.target.id !== 'enviar') resetarTimer();
  });
}
iniciarMonitoramentoInatividade();

// 🔒 Validação da chave de acesso
async function validarChave() {
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

  try {
    const resposta = await fetch("https://learningbyworking.app.n8n.cloud/webhook/senha", {
      method: "POST",
      body: formData
    });

    if (!resposta.ok) throw new Error("Servidor não respondeu corretamente");

    const json = await resposta.json();
    if (json && ((Array.isArray(json) && json.length > 0 && json[0].nome) || (json.nome))) {
      const nome = Array.isArray(json) ? json[0].nome : json.nome;
      msg.textContent = `✅ Acesso aprovado! Bem-vindo, ${nome}!`;
      msg.style.color = "green";

      ['prompt', 'arquivo', 'enviar', 'remover', 'ferramenta', 'grafico_tipo', 'coluna_y', 'colunas_x'].forEach(id => {
        document.getElementById(id).disabled = false;
      });

      document.getElementById('logout').style.display = 'inline-block';

      const campoSenha = document.getElementById('chave');
      campoSenha.value = '';
      campoSenha.required = false;
      campoSenha.disabled = true;

      if (slimSelectInstance) slimSelectInstance.destroy();
      slimSelectInstance = new SlimSelect({
        select: '#colunas_x',
        settings: { closeOnSelect: false }
      });

      sessaoAtiva = true;
      resetarTimer();
    } else {
      msg.textContent = "❌ Chave incorreta ou sem autorização.";
      msg.style.color = "red";
    }
  } catch (err) {
    msg.textContent = "❌ Erro na verificação.";
    msg.style.color = "red";
  } finally {
    spinner.style.display = "none";
  }
}

// 🧼 Deslogar
function deslogar() {
  if (!confirm("Tem certeza que deseja sair?\nTudo será apagado.")) return;

  sessaoAtiva = false;
  clearTimeout(inatividadeTimer);

  ['prompt', 'arquivo', 'enviar', 'remover', 'ferramenta', 'grafico_tipo', 'coluna_y', 'colunas_x'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.disabled = true;
      if (el.tagName === "SELECT" || el.tagName === "INPUT" || el.tagName === "TEXTAREA") el.value = '';
    }
  });

  if (slimSelectInstance) {
    slimSelectInstance.destroy();
    slimSelectInstance = null;
  }

  document.getElementById('analise').innerHTML = '';
  document.getElementById('grafico').innerHTML = '';
  document.getElementById('erro-arquivo').textContent = '';
  const msg = document.getElementById('mensagem-chave');
  msg.textContent = '👋 Até a próxima!';
  msg.style.color = 'gray';

  setTimeout(() => { msg.textContent = ''; }, 2000);

  const campoSenha = document.getElementById('chave');
  const btnValidar = document.getElementById('validar');
  campoSenha.required = true;
  campoSenha.disabled = false;
  campoSenha.value = '';
  btnValidar.disabled = false;
  btnValidar.textContent = 'Validar';
  document.getElementById('logout').style.display = 'none';
}






