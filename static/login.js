// CrediLens Login/Signup Panel Toggle
document.addEventListener('DOMContentLoaded', function() {
    const signInBtn = document.getElementById("signIn");
    const signUpBtn = document.getElementById("signUp");
    const container = document.querySelector(".container");
    
    if (signInBtn && signUpBtn && container) {
        // Toggle to Consumer panel (left side)
        signInBtn.addEventListener("click", () => {
            container.classList.remove("right-panel-active");
        });

        // Toggle to Producer panel (right side)
        signUpBtn.addEventListener("click", () => {
            container.classList.add("right-panel-active");
        });
    }
    
    // Auto-hide notifications after 5 seconds
    const notification = document.querySelector('.notification');
    if (notification) {
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transition = 'opacity 0.5s ease';
            setTimeout(() => {
                notification.remove();
            }, 500);
        }, 5000);
    }
});
