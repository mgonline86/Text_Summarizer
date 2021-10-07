const textBox = document.querySelector('.try-sec textarea.txt');
const count = document.querySelector('.try-sec .counter .value');

let countLetters = () => {
    const text = textBox.value;
    const textLength = text.length;

    count.innerHTML = textLength;
}

textBox.addEventListener('keyup', countLetters);