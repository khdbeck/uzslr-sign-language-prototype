const search = document.getElementById('wordSearch');
const list = document.getElementById('wordList');
const count = document.getElementById('wordCount');
let entries = [];

function exampleVideo(example, word) {
  // Poster only until hovered or tapped; preload="none" keeps 50 clips off the network.
  const video = document.createElement('video');
  Object.assign(video, { src: example.video, poster: example.poster, muted: true, loop: true, playsInline: true, preload: 'none' });
  video.className = 'slovo-word-video';
  video.setAttribute('aria-label', `Ishora namunasi: ${word}`);
  video.addEventListener('mouseenter', () => video.play().catch(() => {}));
  video.addEventListener('mouseleave', () => video.pause());
  video.addEventListener('click', () => (video.paused ? video.play().catch(() => {}) : video.pause()));
  return video;
}

function renderWords() {
  const query = search.value.trim().toLocaleLowerCase('ru');
  const matches = entries.filter(({ word }) => word.toLocaleLowerCase('ru').includes(query));
  count.textContent = `${matches.length} / ${entries.length} ta ishora`;
  list.replaceChildren(...matches.map(({ word, example }) => {
    const item = document.createElement('li');
    if (example) {
      item.classList.add('has-example');
      item.append(exampleVideo(example, word));
    }
    const label = document.createElement('span');
    label.textContent = word;
    item.append(label);
    return item;
  }));
}
search.addEventListener('input', renderWords);
fetch('/api/recognition-signs').then(async (response) => {
  if (!response.ok) throw new Error('Lug‘atni yuklab bo‘lmadi');
  const data = await response.json();
  entries = data.signs.map((word, i) => ({ word, example: data.examples?.[i] ?? null }));
  search.value = decodeURIComponent(location.hash.slice(1));
  renderWords();
}).catch((error) => { count.textContent = error.message; });
