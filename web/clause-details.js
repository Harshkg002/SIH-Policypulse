async function loadComments() {
  // Get clause ID
  const urlParams = new URLSearchParams(window.location.search);
  const provision =
    urlParams.get("provision") || localStorage.getItem("selectedProvision");
  document.getElementById(
    "page-title"
  ).textContent = `Comments for ${provision}`;

  // Fetch all individual comments
  const response = await fetch("individual_comments.json");
  const allComments = await response.json();

  // Filter for this provision
  let comments = allComments.filter((c) => c.provision_number === provision);

  const container = document.getElementById("comments-container");
  const filterContainer = document.getElementById("stakeholderFilter");
  container.innerHTML = "";
  filterContainer.innerHTML = "";

  if (!comments.length) {
    container.innerHTML = `<p class="placeholder">No comments found for ${provision}</p>`;
    return;
  }

  // --- Create filter dropdown in topbar ---
  const stakeholders = [...new Set(comments.map((c) => c.stakeholder_type))];

  const label = document.createElement("label");
  label.textContent = "Filter: ";

  const select = document.createElement("select");
  select.innerHTML = `<option value="all">All</option>`;
  stakeholders.forEach((type) => {
    const option = document.createElement("option");
    option.value = type;
    option.textContent = type;
    select.appendChild(option);
  });

  filterContainer.appendChild(label);
  filterContainer.appendChild(select);

  // --- Render cards helper ---
  function renderCards(list) {
    // Clear old cards
    container.innerHTML = "";

    if (!list.length) {
      const msg = document.createElement("p");
      msg.className = "placeholder";
      msg.textContent = "No comments match the filter.";
      container.appendChild(msg);
      return;
    }

    list.forEach((c) => {
      const card = document.createElement("div");
      card.className = "hot-card";
      card.innerHTML = `
        <div class="title">
          <strong>${c.stakeholder_type}</strong>
          <div style="flex:1"></div>
          <span class="tag">${c.sentiment}</span>
        </div>
        <div class="stat">Controversy Score: ${c.ai_controversy_score}</div>
        <p>${c.comment_text}</p>
        <div class="key-concerns">
          ${(c.ai_main_concerns || [])
            .map(
              (concern) => `<span class="concern-pill">${concern}</span>`
            )
            .join("")}
        </div>
      `;
      container.appendChild(card);
    });
  }

  // Initial render
  renderCards(comments);

  // Handle filter change
  select.addEventListener("change", () => {
    const value = select.value;
    if (value === "all") {
      renderCards(comments);
    } else {
      renderCards(comments.filter((c) => c.stakeholder_type === value));
    }
  });
}

document.addEventListener("DOMContentLoaded", loadComments);
