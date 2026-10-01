const form = document.querySelector('#predict-form');
const result = document.querySelector('#result');
const errorBox = document.querySelector('#error');
const categories = ['Budget', 'Mid-range', 'Premium', 'Flagship'];

document.querySelectorAll('.range').forEach(range => {
  const number = form.elements[range.dataset.for];
  range.addEventListener('input', () => number.value = range.value);
  number.addEventListener('input', () => range.value = number.value);
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  errorBox.hidden = true;
  const button = form.querySelector('.submit-btn');
  button.disabled = true;
  button.innerHTML = 'Reading your specs…';
  const data = new FormData(form);
  const payload = Object.fromEntries(data.entries());
  ['ram', 'battery', 'screen', 'cores', 'storage', 'camera', 'resolution'].forEach(key => payload[key] = Number(payload[key]));
  payload.five_g = form.elements.five_g.checked;
  try {
    const response = await fetch('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Could not estimate this device.');
    document.querySelector('#category').textContent = body.category;
    document.querySelector('#band').textContent = body.band;
    document.querySelector('#description').textContent = body.description;
    document.querySelector('#confidence').textContent = `${Math.round(body.confidence * 100)}%`;
    const chart = document.querySelector('#probabilities');
    chart.replaceChildren(...categories.map(category => {
      const value = body.probabilities[category] || 0;
      const row = document.createElement('div');
      row.className = `prob-row${category === body.category ? ' active' : ''}`;
      row.innerHTML = `<span>${category}</span><div class="prob-track"><div class="prob-fill" style="width:${Math.round(value * 100)}%"></div></div><b>${Math.round(value * 100)}%</b>`;
      return row;
    }));
    result.hidden = false;
    result.scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
    button.innerHTML = 'Estimate my phone <span>↗</span>';
  }
});

document.querySelector('#reset').addEventListener('click', () => {
  result.hidden = true;
  document.querySelector('#estimator').scrollIntoView({behavior: 'smooth'});
});
