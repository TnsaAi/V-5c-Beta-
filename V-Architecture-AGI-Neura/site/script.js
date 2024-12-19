document.addEventListener("DOMContentLoaded", () => {
    const navLinks = document.querySelector('.nav-links');
    const burgerMenu = document.querySelector('.burger-menu');

    burgerMenu.addEventListener('click', () => {
        navLinks.classList.toggle('active');
    });
});
