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

// ✅ Validação dos campos do formulário
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

