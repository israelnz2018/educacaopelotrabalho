let sessaoAtiva = false;
let inatividadeTimer = null;
let slimSelectInstance = null;

// ⏱️ Timer de inatividade
function iniciarContadorInatividade() {
  clearTimeout(inatividadeTimer);
  inatividadeTimer = setTimeout(deslogar, 10 * 60 * 1000);
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
  msg.style.color = 'black';

  document.getElementById('chave').required = true;
  document.getElementById('chave').disabled = false;
  document.getElementById('chave').value = '';
  document.getElementById('validar').disabled = false;
  document.getElementById('validar').textContent = 'Validar';
  document.getElementById('sair').style.display = 'none';
}

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

    const raw = await resposta.json();
    const data = Array.isArray(raw) ? raw : [raw];
    const itemComNome = data.find(item => item?.nome?.trim());

    if (itemComNome) {
      msg.textContent = `✅ Acesso aprovado! Bem-vindo, ${itemComNome.nome}!`;
      msg.style.color = "green";

      ['prompt', 'arquivo', 'enviar', 'remover', 'ferramenta', 'grafico_tipo', 'coluna_y', 'colunas_x'].forEach(id => {
        document.getElementById(id).disabled = false;
      });

      document.getElementById('sair').style.display = 'inline-block';

      document.getElementById('chave').value = '';
      document.getElementById('chave').required = false;
      document.getElementById('chave').disabled = true;

      if (slimSelectInstance) slimSelectInstance.destroy();
      slimSelectInstance = new SlimSelect({
        select: '#colunas_x',
        settings: { closeOnSelect: false }
      });

      sessaoAtiva = true;
      iniciarContadorInatividade();
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

// ❌ Modal erro
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

// ✅ Validação (ajustada para select múltiplo)
function validarCamposSelecionados() {
  const ferramenta = document.getElementById('ferramenta').value;
  const grafico = document.getElementById('grafico_tipo').value;
  const colunaY = document.getElementById('coluna_y').value;
  const colunasX = Array.from(document.getElementById('colunas_x').selectedOptions).map(opt => opt.value);

  if (!ferramenta && !grafico) {
    mostrarModalErro("Escolha uma análise ou um gráfico.");
    return false;
  }

  if (!colunaY && colunasX.length === 0) {
    mostrarModalErro("Selecione pelo menos uma coluna.");
    return false;
  }

  return true;
}

// 🚀 Enviar formulário
async function enviarFormulario(event) {
  event.preventDefault();

  if (!validarCamposSelecionados()) return;

  const spinner = document.getElementById('spinner');
  spinner.style.display = "inline-block";

  const form = document.getElementById('formulario');
  const formData = new FormData(form);
  formData.delete('senha');

  const containerAnalise = document.getElementById('analise');
  const carregandoExistente = document.getElementById('carregando-analise');
  if (carregandoExistente) carregandoExistente.remove();

  const carregando = document.createElement('div');
  carregando.id = "carregando-analise";
  carregando.textContent = "Processando...";
  carregando.style.marginBottom = "12px";
  containerAnalise.prepend(carregando);

  try {
    const resposta = await fetch("https://webhook.site/...", {
      method: "POST",
      body: formData
    });

    const resultado = await resposta.json();
    spinner.style.display = "none";
    carregando.remove();

    if (resultado.erro) {
      mostrarModalErro(resultado.erro);
    } else {
      if (resultado.analise) {
        document.getElementById("analise").innerHTML = resultado.analise;
      }
      if (resultado.grafico_base64) {
        document.getElementById("grafico").innerHTML = `<img src="data:image/png;base64,${resultado.grafico_base64}" alt="Gráfico">`;
      }
    }

  } catch (erro) {
    spinner.style.display = "none";
    carregando.remove();
    mostrarModalErro("Erro ao enviar os dados.");
  }
}


