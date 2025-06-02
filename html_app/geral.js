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

// 🧹 Remover arquivo
function removerArquivo() {
  document.getElementById('arquivo').value = '';
  document.getElementById('erro-arquivo').textContent = '';
}

// 📁 Verificar arquivo
function verificarArquivo() {
  const arquivo = document.getElementById('arquivo').files[0];
  const erro = document.getElementById('erro-arquivo');
  if (!arquivo) return;

  const tipoAceito = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';

  if (arquivo.type !== tipoAceito && !arquivo.name.endsWith('.xlsx')) {
    erro.textContent = "❌ Formato não aceito. Envie apenas arquivos .xlsx do Excel.";
    document.getElementById('arquivo').value = '';
  } else {
    erro.textContent = '';
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
    const resposta = await fetch('https://learningbyworking.app.n8n.cloud/webhook-test/Agente-gpt', {
      method: 'POST',
      body: formData
    });

    if (!resposta.ok) throw new Error("Servidor não respondeu corretamente");

    const json = await resposta.json();

    if (json.grafico_isolado_base64) {
      const novoGrafico = `
        <div class="grafico-isolado" style="margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid #ddd;">
          <img src="data:image/png;base64,${json.grafico_isolado_base64}" style="max-width: 100%; border-radius: 6px;" alt="Gráfico isolado" />
        </div>
      `;
      const containerGrafico = document.getElementById('grafico');
      containerGrafico.insertAdjacentHTML('afterbegin', novoGrafico);
    }

    if (json.analise && json.analise.trim() !== "") {
      const carregandoAntigo = document.getElementById('carregando-analise');
      if (carregandoAntigo) carregandoAntigo.remove();

      const parteAnalise = json.analise
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');

      const novaAnalise = `
        <div class="analise-completa" style="margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid #ccc;">
          <div class="analise-texto">${parteAnalise}</div>
          ${json.grafico_base64 ? `<div class="analise-graficos"><img src="data:image/png;base64,${json.grafico_base64}" alt="Gráfico da análise" style="margin-top: 12px;" /></div>` : ""}
        </div>
      `;
      containerAnalise.insertAdjacentHTML('afterbegin', novaAnalise);
    }

  } catch (error) {
    const erroMsg = document.createElement('div');
    erroMsg.style.color = 'red';
    erroMsg.style.marginBottom = '12px';
    erroMsg.textContent = `❌ Erro ao processar resposta: ${error.message}`;
    containerAnalise.insertBefore(erroMsg, containerAnalise.firstChild);
  } finally {
    spinner.style.display = "none";
  }
}





