  // app.js — updated to load external JSON files

  // ---------- DOM references ----------
  const stakeholderSelect = document.getElementById("stakeholderFilter");
  const worklistTableBody = document.querySelector("#worklistTable tbody");
  const hotClausesContainer = document.getElementById("hot-clauses");
  const inboxContent = document.getElementById("inboxContent");
  const exportCsvBtn = document.getElementById("exportCsvBtn");
  const exportPdfBtn = document.getElementById("exportPdfBtn");

  let clauses = [];
  let individualComments = [];
  let sentimentChart = null;
  let wordCloudData = null;

  // ---------- Utilities ----------
  function controversyScore(c){
    return c.controversy_score || (c.comment_volume * (c.negative_percentage/100));
  }

  function formatPct(n){ return (n*100).toFixed(0) + "%"; }

  // ---------- Renderers ----------
  function renderHotClauses(filteredClauses){
    const top3 = [...filteredClauses].sort((a,b)=>controversyScore(b)-controversyScore(a)).slice(0,3);
    hotClausesContainer.innerHTML = "";
    top3.forEach(c=>{
      const negPct = formatPct((c.negative_comments||0)/(c.comment_volume||1));
      const card = document.createElement("div");
      card.className = "hot-card";
      card.innerHTML = `
        <div class="title"><span class="tag">🔥 Hot</span><div style="flex:1"></div><div class="tag">${c.provision_number}</div></div>
        <div class="stat">${negPct} Negative • ${c.comment_volume} Comments</div>
        <div>Key Concern: <strong>${(c.top_concerns||[])[0] || "—"}</strong></div>
        <div style="display:flex;gap:8px;margin-top:8px">
          <button class="btn-view" data-provision="${c.provision_number}">View Details</button>
          <div style="align-self:center;color:var(--muted);font-size:13px">Controversy: <strong>${controversyScore(c).toFixed(1)}</strong></div>
        </div>
      `;
      hotClausesContainer.appendChild(card);
    });
  }

  function renderTable(filteredClauses){
    const rows = [...filteredClauses].sort((a,b)=>controversyScore(b)-controversyScore(a));
    worklistTableBody.innerHTML = "";
    rows.forEach(c=>{
      const negPct = Math.round(c.negative_percentage||0);
      const neuPct = Math.round(100 - (c.negative_percentage||0) - (c.positive_percentage||0));
      const posPct = Math.round(c.positive_percentage||0);

      const miniBar = `
        <div class="mini-bar" title="Neg:${negPct}% Neu:${neuPct}% Pos:${posPct}%">
          <i class="neg" style="width:${negPct}%"></i>
          <i class="neu" style="width:${neuPct}%"></i>
          <i class="pos" style="width:${posPct}%"></i>
        </div>
      `;
      const concernsHtml = (c.top_concerns||[]).slice(0,3).map(k=>`<span class="concern-pill">${k}</span>`).join("");
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${c.provision_number}</strong></td>
        <td>${c.comment_volume}</td>
        <td>${miniBar}</td>
        <td><div class="key-concerns">${concernsHtml}</div></td>
        <td>${controversyScore(c).toFixed(1)}</td>
        <td><button class="btn-view" data-provision="${c.provision_number}">View Details</button></td>
      `;
      worklistTableBody.appendChild(tr);
    });

    // inside renderHotClauses and renderTable, change button listener:
    document.querySelectorAll(".btn-view").forEach(btn => {
      btn.addEventListener("click", e => {
        const provision = e.currentTarget.dataset.provision;
        // store the provision in localStorage for the details page
        localStorage.setItem("selectedProvision", provision);
        window.location.href = `clause-details.html?provision=${encodeURIComponent(provision)}`;
      });
    });
  }

  function renderSentimentChart(filteredClauses){
    const total = filteredClauses.reduce((acc,c)=>({
      pos: acc.pos + (c.positive_comments||0),
      neg: acc.neg + (c.negative_comments||0),
      neu: acc.neu + (c.neutral_comments||0)
    }), {pos:0,neg:0,neu:0});
    const grand = total.pos + total.neg + total.neu || 1;
    const labels = ["Negative","Positive","Neutral"];
    const data = [total.neg/grand, total.pos/grand, total.neu/grand].map(v=>Math.round(v*100));
    const ctx = document.getElementById("sentimentChart").getContext("2d");
    if(sentimentChart) sentimentChart.destroy();
    sentimentChart = new Chart(ctx, {
      type: 'doughnut',
      data:{
        labels: labels,
        datasets:[{
          data: data,
          backgroundColor: ['#ff6b6b','#33d69f','#cbd5e1'],
          borderWidth:0
        }]
      },
      options:{
        plugins:{
          legend:{position:'bottom',labels:{color:'#cbd5e1'}},
          tooltip:{callbacks:{
            label: ctx=>{
              return `${ctx.label}: ${ctx.parsed}%`;
            }
          }}
        },
        elements:{arc:{borderWidth:0}}
      }
    });
  }

  // ---------- Renderers ----------
async function renderWordCloud(){
  if(!wordCloudData) return;

  // Example: use overall frequencies
  const freqData = wordCloudData.overall;

  // Convert object → list format for WordCloud
  const list = Object.entries(freqData).map(([w,c]) => [w, c * 10]);

  const wordcloudElement = document.getElementById("wordcloud");
  WordCloud(wordcloudElement, {
    list,
    gridSize: Math.round(16 * wordcloudElement.offsetWidth / 1024),
    weightFactor: (size) => Math.pow(size,1.1),
    rotateRatio: 0.25,
    rotationSteps: 2,
    backgroundColor: '#0b1220',
    color: function(word){
      if(word.includes('penalty') || word.includes('burden') || word.includes('compliance')) 
        return '#ff6b6b';
      return '#94a3b8';
    }
  });
}


  function openSmartInbox(provisionNumber){
    const provisionComments = individualComments.filter(c=>c.provision_number===provisionNumber);
    if(provisionComments.length === 0){
      inboxContent.innerHTML = `<p>No comments for ${provisionNumber}</p>`;
      return;
    }
    const html = `
      <h3>${provisionNumber} — ${provisionComments.length} comments</h3>
      <ul>
        ${provisionComments.map(c=>`<li>[${c.stakeholder_type}] "${c.comment_text}" — ${c.sentiment}</li>`).join("")}
      </ul>
    `;
    inboxContent.innerHTML = html;
  }

  // ---------- Filtering ----------
  function filterByStakeholder(stakeholder){
    if(stakeholder === "All") return clauses;

    // For each clause, compute filtered comment stats
    return clauses.map(c => {
      // all comments for this clause
      const provComments = individualComments.filter(ic => ic.provision_number === c.provision_number);

      // comments for the selected stakeholder
      const filteredComments = provComments.filter(ic => ic.stakeholder_type === stakeholder);

      if(filteredComments.length === 0) return null; // skip clauses with no comments for this stakeholder

      // compute percentages and counts
      const total = filteredComments.length;
      const neg = filteredComments.filter(ic => ic.sentiment === "NEGATIVE").length;
      const pos = filteredComments.filter(ic => ic.sentiment === "POSITIVE").length;
      const neu = filteredComments.filter(ic => ic.sentiment === "NEUTRAL").length;
      const topConcerns = []; // optionally recompute from filteredComments if needed

      return {
        provision_number: c.provision_number,
        comment_volume: total,
        negative_comments: neg,
        positive_comments: pos,
        neutral_comments: neu,
        negative_percentage: (neg/total)*100,
        positive_percentage: (pos/total)*100,
        controversy_score: filteredComments.reduce((acc, ic) => acc + (ic.ai_controversy_score||0),0),
        top_concerns: c.top_concerns,
        main_themes: c.main_themes
      };
    }).filter(c => c !== null); // remove clauses with no comments for this stakeholder
  }


  // ---------- Export functions ----------
  function downloadCSV(rows, filename='priority_worklist.csv'){
    const headers = ['Clause','Total Comments','Pct Negative','Pct Positive','Pct Neutral','Key Concerns','ControversyScore'];
    const lines = [headers.join(',')];
    rows.forEach(c=>{
      const line = [
        `"${c.provision_number}"`,
        c.comment_volume,
        (c.negative_percentage||0).toFixed(3),
        (c.positive_percentage||0).toFixed(3),
        (c.neutral_comments||0).toFixed(3),
        `"${(c.top_concerns||[]).slice(0,3).join(';')}"`,
        controversyScore(c).toFixed(3)
      ].join(',');
      lines.push(line);
    });
    const blob = new Blob([lines.join('\n')], {type:'text/csv;charset=utf-8;'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

  async function exportPdf(rows){
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({unit:'pt', format:'a4'});
    doc.setFontSize(14);
    doc.text('Priority Worklist Report', 40, 50);
    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 40, 66);
    let y = 90;
    doc.setFontSize(9);
    doc.text('Clause', 40, y); doc.text('Comments',140,y); doc.text('Neg%',200,y); doc.text('Controversy',260,y);
    y += 12;
    rows.forEach(c=>{
      if(y>750){ doc.addPage(); y=40; }
      doc.text(c.provision_number,40,y);
      doc.text(String(c.comment_volume),140,y);
      doc.text(formatPct((c.negative_percentage||0)/100),200,y);
      doc.text(String(controversyScore(c).toFixed(1)),260,y);
      y += 12;
    });
    doc.save('priority_worklist_report.pdf');
  }

  // ---------- Main update ----------
  function updateDashboard(){
    const stakeholder = stakeholderSelect.value;
    const filtered = filterByStakeholder(stakeholder);
    renderHotClauses(filtered);
    renderTable(filtered);
    renderSentimentChart(filtered);
    renderWordCloud();
  }

  // ---------- Events ----------
  stakeholderSelect.addEventListener("change", ()=> updateDashboard());
  exportCsvBtn.addEventListener("click", ()=>{
    const filtered = filterByStakeholder(stakeholderSelect.value);
    downloadCSV(filtered);
  });
  exportPdfBtn.addEventListener("click", ()=>{
    const filtered = filterByStakeholder(stakeholderSelect.value);
    exportPdf(filtered);
  });

  // ---------- Fetch data ----------
  async function loadData(){
    const provResp = await fetch("provision_summary.json");
    const indResp = await fetch("individual_comments.json");
    const wcResp = await fetch("word_cloud_data.json");
    const provJson = await provResp.json();
    const indJson = await indResp.json();
    const wcJson = await wcResp.json();

    clauses = provJson.provision_analysis || [];
    individualComments = indJson || [];
      wordCloudData = wcJson || {}; 

    updateDashboard();
  }

  loadData();
