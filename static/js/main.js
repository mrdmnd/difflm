document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const genLengthSlider = document.getElementById('gen-length-slider');
    const genLengthValue = document.getElementById('gen-length-value');
    const stepsSlider = document.getElementById('steps-slider');
    const stepsValue = document.getElementById('steps-value');
    const tempSlider = document.getElementById('temp-slider');
    const tempValue = document.getElementById('temp-value');
    const promptInput = document.getElementById('prompt-input');
    const playBtn = document.getElementById('play-btn');
    const stepBtn = document.getElementById('step-btn');
    const resetBtn = document.getElementById('reset-btn');
    const scheduleVis = document.getElementById('schedule-vis');
    const canvasDisplay = document.getElementById('canvas-display');
    const tokenProbVis = document.getElementById('token-prob-vis');

    // --- State ---
    let state = null;
    let isPlaying = false;
    let playInterval = null;

    // --- UI Update Functions ---
    function updateSliderValue(slider, valueEl) {
        valueEl.textContent = slider.value;
    }

    function updateControlsState() {
        const hasPrompt = promptInput.value.trim() !== '';
        const hasState = state !== null;

        // Play and Step can start a generation if there's a prompt,
        // or continue one if state exists.
        playBtn.disabled = !hasPrompt && !hasState;
        stepBtn.disabled = !hasPrompt && !hasState;

        // Reset only makes sense if there is state.
        resetBtn.disabled = !hasState;

        // Special case: if generation is finished, disable play/step.
        if (hasState && state.current_step >= state.max_steps - 1) {
            stopPlayback();
            playBtn.disabled = true;
            stepBtn.disabled = true;
        }
    }

    // --- D3 Visualizations ---
    function drawSchedule(schedule) {
        if (!schedule || schedule.length === 0) {
            scheduleVis.innerHTML = '<p style="text-align: center;">No schedule data.</p>';
            return;
        }
        const svg = d3.select(scheduleVis).html("").append("svg")
            .attr("width", "100%")
            .attr("height", "100%");

        const margin = { top: 20, right: 20, bottom: 40, left: 50 };
        const width = scheduleVis.clientWidth - margin.left - margin.right;
        const height = scheduleVis.clientHeight - margin.top - margin.bottom;

        const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

        const x = d3.scaleBand()
            .rangeRound([0, width])
            .padding(0.1)
            .domain(schedule.map((d, i) => i));

        const y = d3.scaleLinear()
            .rangeRound([height, 0])
            .domain([0, d3.max(schedule)]);

        g.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(x).tickValues(x.domain().filter((d, i) => i % 5 === 0 || i === schedule.length - 1)))
            .append("text")
            .attr("fill", "#fff")
            .attr("x", width / 2)
            .attr("y", margin.bottom - 10)
            .attr("text-anchor", "middle")
            .text("Diffusion Step");

        g.append("g")
            .call(d3.axisLeft(y))
            .append("text")
            .attr("fill", "#fff")
            .attr("transform", "rotate(-90)")
            .attr("y", 6)
            .attr("dy", "-3.5em")
            .attr("text-anchor", "end")
            .text("Tokens to Sample");

        g.selectAll(".bar")
            .data(schedule)
            .enter().append("rect")
            .attr("class", "bar")
            .attr("x", (d, i) => x(i))
            .attr("y", d => y(d))
            .attr("width", x.bandwidth())
            .attr("height", d => height - y(d))
            .attr("fill", (d, i) => (state && i === state.current_step) ? "#28a745" : "#4267B2");

        // Highlight current step
        if (state) {
            g.selectAll(".bar")
                .filter((d, i) => i === state.current_step)
                .attr("fill", "#28a745");
        }
    }

    function renderCanvas() {
        canvasDisplay.innerHTML = "";
        if (!state) return;

        const { decoded_tokens, prompt_length } = state;
        decoded_tokens.forEach((token, index) => {
            const span = document.createElement('span');
            // Make whitespace visible
            const formattedToken = token.replace(/\\n/g, '↵\n').replace(/ /g, '␣');
            span.textContent = formattedToken;
            span.classList.add('token');
            span.dataset.index = index;

            if (index < prompt_length) {
                span.classList.add('prompt-token');
            } else if (token === '[MASK]') {
                span.classList.add('mask-token');
            } else {
                span.classList.add('generated-token');
            }
            span.addEventListener('mouseover', handleTokenMouseover);
            canvasDisplay.appendChild(span);
        });
    }

    function drawTokenProbabilities(data) {
        if (!data) {
            tokenProbVis.innerHTML = '';
            return;
        }
        const [tokens, probabilities] = data;
        const svg = d3.select(tokenProbVis).html("").append("svg")
            .attr("width", "100%")
            .attr("height", "100%");

        const margin = { top: 5, right: 20, bottom: 40, left: 60 };
        const width = tokenProbVis.clientWidth - margin.left - margin.right;
        const height = tokenProbVis.clientHeight - margin.top - margin.bottom;

        const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear().range([0, width]).domain([0, d3.max(probabilities)]);
        const y = d3.scaleBand().range([height, 0]).padding(0.1).domain(tokens.map(t => t.replace(/\\n/g, '↵').replace(/ /g, '␣')));

        g.append("g").call(d3.axisLeft(y));
        g.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x).ticks(3, "%"));

        g.selectAll(".bar")
            .data(probabilities)
            .enter().append("rect")
            .attr("class", "bar")
            .attr("y", (d, i) => y(tokens[i].replace(/\\n/g, '↵').replace(/ /g, '␣')))
            .attr("height", y.bandwidth())
            .attr("x", 0)
            .attr("width", d => x(d))
            .attr("fill", "#6BAA75");
    }


    // --- API Communication ---
    async function createStateAndFetch() {
        const prompt = promptInput.value;
        if (!prompt) {
            alert("Please enter a prompt.");
            return;
        }

        stopPlayback(); // Stop any existing playback

        try {
            const response = await fetch('/api/create_state', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    generation_length: parseInt(genLengthSlider.value, 10),
                    steps: parseInt(stepsSlider.value, 10),
                    sampling_temperature: parseFloat(tempSlider.value)
                }),
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            state = await response.json();
            updateUI();
        } catch (error) {
            console.error("Could not create state:", error);
            alert("Failed to start generation. Is the server running?");
        }
    }

    async function performStep() {
        if (!state) return;

        try {
            const response = await fetch('/api/step', { method: 'POST' });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            state = await response.json();
            updateUI();

            if (state.current_step >= state.max_steps - 1) {
                stopPlayback();
            }
        } catch (error) {
            console.error("Could not perform step:", error);
            stopPlayback();
        }
    }

    async function performReset() {
        stopPlayback();
        if (!state) return;

        try {
            const response = await fetch('/api/reset', { method: 'POST' });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            state = await response.json();
            updateUI();
        } catch (error) {
            console.error("Could not reset:", error);
        }
    }

    async function fetchTokenProbabilities(position) {
        if (!state) return;
        try {
            const response = await fetch(`/api/token_probabilities/${position}`);
            if (!response.ok) throw new Error('Failed to fetch token probabilities');
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            drawTokenProbabilities(data);
        } catch (error) {
            console.error(error);
            tokenProbVis.innerHTML = `<p style="color: red; font-size: 0.8em;">${error.message}</p>`;
        }
    }

    // --- Event Handlers & Control Flow ---
    function handleTokenMouseover(event) {
        const index = parseInt(event.target.dataset.index, 10);
        fetchTokenProbabilities(index);
    }

    function startPlayback() {
        if (isPlaying) return;
        if (!state) {
            // If there's no state, create it first, then play.
            createStateAndFetch().then(() => {
                if (state) {
                    isPlaying = true;
                    playBtn.textContent = "Pause";
                    playInterval = setInterval(performStep, 100);
                }
            });
        } else {
            // If state exists, just start playing from where it is.
            isPlaying = true;
            playBtn.textContent = "Pause";
            playInterval = setInterval(performStep, 100);
        }
    }

    function stopPlayback() {
        if (!isPlaying) return;
        isPlaying = false;
        playBtn.textContent = "Play";
        clearInterval(playInterval);
        playInterval = null;
    }

    function handleStep() {
        stopPlayback(); // Pressing step should stop playback
        if (!state) {
            createStateAndFetch();
        } else {
            performStep();
        }
    }

    function updateUI() {
        if (!state) {
            // Reset UI to initial state
            canvasDisplay.innerHTML = '<p style="text-align: center;">Enter a prompt and start a generation.</p>';
            scheduleVis.innerHTML = '';
            tokenProbVis.innerHTML = '';
        } else {
            renderCanvas();
            drawSchedule(state.schedule);
        }
        updateControlsState();
    }

    function init() {
        // Set up sliders
        updateSliderValue(genLengthSlider, genLengthValue);
        updateSliderValue(stepsSlider, stepsValue);
        updateSliderValue(tempSlider, tempValue);
        genLengthSlider.addEventListener('input', () => updateSliderValue(genLengthSlider, genLengthValue));
        stepsSlider.addEventListener('input', () => updateSliderValue(stepsSlider, stepsValue));
        tempSlider.addEventListener('input', () => updateSliderValue(tempSlider, tempValue));
        promptInput.addEventListener('input', updateControlsState);

        // Set up button listeners
        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                createStateAndFetch();
            }
        });
        playBtn.addEventListener('click', () => {
            if (isPlaying) stopPlayback();
            else startPlayback();
        });
        stepBtn.addEventListener('click', handleStep);
        resetBtn.addEventListener('click', performReset);

        // Initial UI state
        updateUI();
        updateControlsState();
    }

    init();
});
