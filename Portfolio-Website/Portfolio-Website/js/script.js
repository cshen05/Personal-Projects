// Change Navigation Background on Scroll
window.addEventListener('scroll', () => {
    const header = document.querySelector('.sticky-header');
    if (window.scrollY > 50) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
});

// Gradual Color Transition Between Sections
window.addEventListener('scroll', () => {
    const hero = document.querySelector('.hero');
    const about = document.querySelector('.about');
    const scrollPosition = window.scrollY;
    const heroHeight = hero.offsetHeight;

    if (scrollPosition <= heroHeight) {
        const ratio = scrollPosition / heroHeight; // Calculate scroll percentage
        const heroColor = [250, 249, 246]; // RGB for #FAF9F6 (Hero Section Color)
        const aboutColor = [210, 180, 140]; // RGB for #D2B48C (About Section Color)

        // Calculate interpolated color
        const interpolatedColor = heroColor.map((start, index) =>
            Math.round(start + ratio * (aboutColor[index] - start))
        );

        // Apply interpolated color to hero section background
        hero.style.backgroundColor = `rgb(${interpolatedColor.join(',')})`;
    }
});