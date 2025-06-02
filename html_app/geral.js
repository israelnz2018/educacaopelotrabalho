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
      if (el.tagName === "SELECT" || el.tagName === "INPUT") el.value = '';
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
}

// ❌ Modal erro
function mostrarModalErro(msg) {
  document.getElementById("modal-erro-texto").textContent = msg;
  document.getElementById("modal-erro").style.display = "flex";
}

function fecharModalErro() {
  document.getElementById("modal-erro").style.display = "none";
}

// ✅ Validação (simplificada — pode ser expandida conforme as regras)
function validarCamposSelecionados() {
  const ferramenta = document.getElementById('ferramenta').value;
  const grafico = document.getElementById('grafico_tipo').value;
  const colunaY = document.getElementById('coluna_y').value;
  const colunasX = document.getElementById('colunas_x').value;

  if (!ferramenta && !grafico) {
    mostrarModalErro("Escolha uma análise ou um gráfico.");
    return false;
  }

  if (!colunaY && !colunasX) {
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

