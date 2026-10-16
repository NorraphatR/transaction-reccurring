/* per payday: the questions the app asked since the previous payday, and model only vs + questions for this payday */
function paydayPanel(c) {
  const Q = c.questions || [];
  const MOM = { 1: "Moment 1 · first payment to someone new", 3: "Moment 3 · amount varies", 4: "Moment 4 · bill looks overdue", 5: "Moment 5b · small payment, very regular: is it a bill?" };
  $("qcards").innerHTML = Q.length ? Q.map((q) => `<div class="qcard ${q.changed ? "changed" : ""}">
      <div class="qhead"><span class="note">${esc(MOM[q.moment] || "Question")} · asked ${esc(q.asked)}</span>${q.changed ? `<span class="qtag">changed the result</span>` : ""}</div>
      <div class="qtext">${esc(q.text)}</div>
      <div class="qopts">${q.options.map((o) => `<span class="qopt ${o === q.answer ? "on" : ""}">${o === q.answer ? "✓ " : ""}${esc(o)}</span>`).join("")}</div>
      <div class="note">Effect: ${esc(q.effect)}</div></div>`).join("")
    : `<p class="note">The app asked nothing between the previous payday and this one: no first payment of ฿1,000 or more to someone new, and no bill the model was unsure about.</p>`;
  const m = c.money, cnt = (tag) => { const B = c.bills[tag].filter((b) => b.verdict !== "correct: not a bill"); return [B.filter((b) => b.verdict.startsWith("correct")).length, B.length]; };
  const [c0, n0] = cnt("model"), [c1, n1] = cnt("questions");
  const d = (a, b, up) => Math.abs(a - b) < 0.5 ? `<span class="note">same</span>` : `<span class="${(up ? b > a : b < a) ? "up" : "down"}">${(up ? b > a : b < a) ? "better" : "worse"}</span>`;
  const rows = [
    ["Bills fully correct", `${c0} of ${n0}`, `${c1} of ${n1}`, d(c0 / Math.max(1, n0), c1 / Math.max(1, n1), true)],
    ["Upcoming bills reserved", baht(m.upcoming_model), baht(m.upcoming_questions), ""],
    ["Real bills paid until payday", baht(m.real_bills), baht(m.real_bills), ""],
    ["Bill shortfall", baht(m.bill_short_model), baht(m.bill_short_questions), d(m.bill_short_model, m.bill_short_questions, false)],
    ["Reserved above need", baht(m.reserved_extra_model), baht(m.reserved_extra_questions), d(m.reserved_extra_model, m.reserved_extra_questions, false)],
    ["Safe to spend", baht(m.safe_model), baht(m.safe_questions), ""],
    ["Overall (buffer − shortfall)", (m.overall_model < 0 ? "short " : "OK +") + baht(Math.abs(m.overall_model)), (m.overall_questions < 0 ? "short " : "OK +") + baht(Math.abs(m.overall_questions)), d(m.overall_model, m.overall_questions, true)]];
  $("qcompare").innerHTML = `<table><thead><tr><th>${esc(c.id)}</th><th>Model only</th><th>+ questions</th><th></th></tr></thead><tbody>${
    rows.map(([k, a, b, x]) => `<tr><td>${k}</td><td>${a}</td><td>${b}</td><td>${x}</td></tr>`).join("")}</tbody></table>`;
}
