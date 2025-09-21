async function loadComments() {
  // Get clause ID
  const urlParams = new URLSearchParams(window.location.search);
  const provision = urlParams.get("provision") || localStorage.getItem("selectedProvision");
  document.getElementById("page-title").textContent = `Comments for ${provision}`;

  // Fetch all individual comments (same file used on dashboard)
  const response = await fetch("individual_comments.json");
  const allComments = await response.json();

  // Filter for this provision
  const comments = allComments.filter(c => c.provision_number === provision);

  const container = document.getElementById("comments-container");
  container.innerHTML = "";

  if (!comments.length) {
    container.innerHTML = `<p class="placeholder">No comments found for ${provision}</p>`;
    return;
  }

  // Build cards for each comment
  comments.forEach(c => {
    const card = document.createElement("div");
    card.className = "hot-card"; // reuse your theme card class
    card.innerHTML = `
      <div class="title">
        <strong>${c.stakeholder_type}</strong>
        <div style="flex:1"></div>
        <span class="tag">${c.sentiment}</span>
      </div>
      <div class="stat">Controversy Score: ${c.ai_controversy_score}</div>
      <p>${c.comment_text}</p>
      <div class="key-concerns">
        ${(c.ai_main_concerns || []).map(concern => `<span class="concern-pill">${concern}</span>`).join("")}
      </div>
    `;
    container.appendChild(card);
  });
}

document.addEventListener("DOMContentLoaded", loadComments);
