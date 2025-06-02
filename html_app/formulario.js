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

// 🚨 Mostrar modal de erro
function mostrarModalErro(mensagem) {
  let modal = document.getElementById("modal-erro");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "modal-erro";
    modal.style.position = "fixed";
    modal.style.top = "0";
    modal.style.left = "0";
    modal.style.width = "100%";
    modal.style.height = "100%";
    modal.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
    modal.style.display = "flex";
    modal.style.justifyContent = "center";
    modal.style.alignItems = "center";
    modal.style.zIndex = "1000";

    const caixa = document.createElement("div");
    caixa.style.backgroundColor = "white";
    caixa.style.padding = "20px";
    caixa.style.borderRadius = "8px";
    caixa.style.maxWidth = "400px";
    caixa.style.textAlign = "center";
    caixa.innerHTML = `
      <p id="modal-erro-texto" style="margin-bottom: 16px; font-size: 16px;"></p>
      <button onclick="fecharModalErro()" style="padding: 6px 12px;">OK</button>
    `;

    modal.appendChild(caixa);
    document.body.appendChild(modal);
  }

  document.getElementById("modal-erro-texto").textContent = mensagem;
  modal.style.display = "flex";
}

// 🚪 Fechar modal de erro
function fecharModalErro() {
  const modal = document.getElementById("modal-erro");
  if (modal) modal.style.display = "none";
}

// Remove mensagens antigas de erro e alertas
function limparMensagensAnteriores() {
  const mensagensErro = document.querySelectorAll(".erro-aviso, .alerta, .mensagem-erro, .falha");
  mensagensErro.forEach(msg => msg.remove());

  const msgErroFetch = document.getElementById("mensagem-erro-fetch");
  if (msgErroFetch) msgErroFetch.remove();

  // Limpa também mensagens dentro do container de análise para evitar bloqueio visual
  const containerAnalise = document.getElementById('analise');
  if (containerAnalise) {
    const filhos = containerAnalise.querySelectorAll('div.mensagem-erro, div.erro-aviso, div.alerta, div.falha');
    filhos.forEach(filho => filho.remove());
  }
}

// ✅ Validação dos campos do formulário (aqui não bloqueia mais o envio, só mostra o modal)
function validarCamposSelecionados() {
  const ferramenta = document.getElementById('ferramenta').value;
  const grafico = document.getElementById('grafico_tipo').value;
  const colunaY = document.getElementById('coluna_y').value;
  const colunasX = Array.from(document.getElementById('colunas_x').selectedOptions).map(opt => opt.value);

  if (!ferramenta && !grafico) {
    mostrarModalErro("Escolha uma análise ou um gráfico.");
    // return false;  <-- REMOVIDO para não bloquear envio
  }

  if (!colunaY && colunasX.length === 0) {
    mostrarModalErro("Selecione pelo menos uma coluna.");
    // return false;  <-- REMOVIDO para não bloquear envio
  }

  return true; // Sempre retorna true para permitir o envio mesmo com erros
}

// 🚀 Enviar formulário
async function enviarFormulario(event) {
  event.preventDefault();

  limparMensagensAnteriores();

  validarCamposSelecionados(); // mostra modal, mas não bloqueia envio

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




