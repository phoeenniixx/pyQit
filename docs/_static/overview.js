document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("object-filter");
  if (!input) return;
  const rows = document.querySelectorAll("table.object-overview tbody tr");
  input.addEventListener("input", () => {
    const terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    rows.forEach((row) => {
      const text = row.textContent.toLowerCase();
      row.style.display = terms.every((t) => text.includes(t)) ? "" : "none";
    });
  });
});
