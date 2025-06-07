-- wordplay.nvim - A Neovim plugin with real NLP capabilities
-- File structure:
-- lua/letterspread/init.lua (main entry point)
-- python/nlp_anagrams.py
-- python/nlp_poetry.py
-- python/nlp_wordsearch.py

-- ============================================================================
-- lua/wordplay/init.lua
-- ============================================================================

local M = {}

-- Configuration
local config = {
	keymaps = {
		anagrams = "<leader>wa",
		poetry = "<leader>wp",
		wordsearch = "<leader>ws",
	},
	window = {
		border = "rounded",
		relative = "editor",
	},
	poetry = {
		default_type = "haiku",
		use_rhymes = true,
		preserve_sentiment = true,
	},
	wordsearch = {
		grid_size_min = 15,
		grid_size_max = 25,
		max_words = 20,
		use_semantic_groups = true,
	},
	anagrams = {
		min_word_length = 3,
		include_semantic_similarity = true,
		filter_by_frequency = true,
	},
	python_path = "python3", -- Can be customized
}

local plugin_root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h")

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

local function get_python_script_path(script_name)
	return plugin_root .. "/python/" .. script_name
end

local function get_buffer_text()
	local lines = vim.api.nvim_buf_get_lines(0, 0, -1, false)
	return table.concat(lines, "\n")
end

local function run_python_script(script_name, input_text, args)
	args = args or {}
	local script_path = get_python_script_path(script_name)

	-- Create temporary file for input
	local temp_input = vim.fn.tempname()
	local temp_output = vim.fn.tempname()

	-- Write input to temp file
	local input_file = io.open(temp_input, "w")
	if not input_file then
		vim.notify("Error: Cannot create temporary input file", vim.log.levels.ERROR)
		return nil
	end
	input_file:write(input_text)
	input_file:close()

	-- Build command
	local cmd = {
		config.python_path,
		script_path,
		temp_input,
		temp_output,
	}

	-- Add additional arguments
	for _, arg in ipairs(args) do
		table.insert(cmd, arg)
	end

	-- Execute command
	local result = vim.fn.system(cmd)
	local exit_code = vim.v.shell_error

	-- Clean up input file
	vim.fn.delete(temp_input)

	if exit_code ~= 0 then
		vim.notify("Python script error: " .. result, vim.log.levels.ERROR)
		vim.fn.delete(temp_output)
		return nil
	end

	-- Read output
	local output_file = io.open(temp_output, "r")
	if not output_file then
		vim.notify("Error: Cannot read output file", vim.log.levels.ERROR)
		return nil
	end

	local output = output_file:read("*all")
	output_file:close()
	vim.fn.delete(temp_output)

	-- Parse JSON output
	local ok, parsed = pcall(vim.json.decode, output)
	if not ok then
		vim.notify("Error parsing Python output: " .. output, vim.log.levels.ERROR)
		return nil
	end

	return parsed
end

local function check_dependencies()
	-- Check if Python is available
	local python_check = vim.fn.system(config.python_path .. " --version")
	if vim.v.shell_error ~= 0 then
		vim.notify("Python not found. Please install Python 3.7+ and ensure it's in PATH", vim.log.levels.ERROR)
		return false
	end

	-- Check if required Python packages are installed
	local required_packages = { "nltk", "spacy", "pyphen", "pronouncing", "numpy" }
	local missing_packages = {}

	for _, package in ipairs(required_packages) do
		local check_cmd = config.python_path .. ' -c "import ' .. package .. '"'
		local result = vim.fn.system(check_cmd)
		if vim.v.shell_error ~= 0 then
			table.insert(missing_packages, package)
		end
	end

	if #missing_packages > 0 then
		local install_cmd = "pip install " .. table.concat(missing_packages, " ")
		vim.notify(
			"Missing Python packages: " .. table.concat(missing_packages, ", ") .. "\nRun: " .. install_cmd,
			vim.log.levels.WARN
		)
		return false
	end

	return true
end

-- ============================================================================
-- ANAGRAM FUNCTIONALITY WITH NLP
-- ============================================================================

local function display_anagrams()
	local text = get_buffer_text()
	if text:match("^%s*$") then
		vim.notify("Buffer is empty or contains only whitespace", vim.log.levels.WARN)
		return
	end

	if not check_dependencies() then
		return
	end

	local args = {
		"--min-length",
		tostring(config.anagrams.min_word_length),
		"--semantic-similarity",
		config.anagrams.include_semantic_similarity and "true" or "false",
		"--filter-frequency",
		config.anagrams.filter_by_frequency and "true" or "false",
	}

	local result = run_python_script("nlp_anagrams.py", text, args)
	if not result then
		return
	end

	if not result.anagram_groups or #result.anagram_groups == 0 then
		vim.notify("No anagrams found in buffer", vim.log.levels.INFO)
		return
	end

	-- Create display buffer
	local buf = vim.api.nvim_create_buf(false, true)
	local lines = {
		"Found " .. #result.anagram_groups .. " anagram groups:",
		"",
	}

	for i, group in ipairs(result.anagram_groups) do
		local line = i .. ". " .. table.concat(group.words, " ↔ ")
		if group.semantic_score then
			line = line .. string.format(" (similarity: %.2f)", group.semantic_score)
		end
		table.insert(lines, line)

		if group.definitions and #group.definitions > 0 then
			table.insert(lines, "   Meanings: " .. table.concat(group.definitions, ", "))
		end
		table.insert(lines, "")
	end

	table.insert(lines, "Press 'q' to close")

	local win = vim.api.nvim_open_win(buf, true, {
		relative = config.window.relative,
		width = 80,
		height = math.min(25, #lines + 2),
		col = 10,
		row = 5,
		border = config.window.border,
		title = "Enhanced Anagrams",
	})

	vim.api.nvim_buf_set_lines(buf, 0, -1, false, lines)
	vim.bo[buf].modifiable = false
	vim.bo[buf].buftype = "nofile"
	vim.keymap.set("n", "q", "<cmd>close<cr>", { buffer = buf, nowait = true })
end

-- ============================================================================
-- POETRY FUNCTIONALITY WITH NLP
-- ============================================================================

local function display_poetry(type)
	type = type or config.poetry.default_type

	local text = get_buffer_text()
	if text:match("^%s*$") then
		vim.notify("Buffer is empty or contains only whitespace", vim.log.levels.WARN)
		return
	end

	if not check_dependencies() then
		return
	end

	local args = {
		"--type",
		type,
		"--use-rhymes",
		config.poetry.use_rhymes and "true" or "false",
		"--preserve-sentiment",
		config.poetry.preserve_sentiment and "true" or "false",
	}

	local result = run_python_script("nlp_poetry.py", text, args)
	if not result then
		return
	end

	if not result.poem or #result.poem == 0 then
		vim.notify("Could not generate " .. type .. " from buffer content", vim.log.levels.WARN)
		return
	end

	-- Create display buffer
	local buf = vim.api.nvim_create_buf(false, true)
	local lines = { "" }

	for _, line in ipairs(result.poem) do
		table.insert(lines, "  " .. line)
	end

	table.insert(lines, "")

	if result.metadata then
		if result.metadata.rhyme_scheme then
			table.insert(lines, "Rhyme scheme: " .. result.metadata.rhyme_scheme)
		end
		if result.metadata.sentiment then
			table.insert(
				lines,
				string.format("Sentiment: %s (%.2f)", result.metadata.sentiment.label, result.metadata.sentiment.score)
			)
		end
		if result.metadata.syllable_counts then
			table.insert(lines, "Syllables: " .. table.concat(result.metadata.syllable_counts, "-"))
		end
		table.insert(lines, "")
	end

	table.insert(lines, "Press 'q' to close")

	local title = "Generated " .. type:sub(1, 1):upper() .. type:sub(2)
	local win = vim.api.nvim_open_win(buf, true, {
		relative = config.window.relative,
		width = 60,
		height = math.min(20, #lines + 2),
		col = 15,
		row = 8,
		border = config.window.border,
		title = title,
	})

	vim.api.nvim_buf_set_lines(buf, 0, -1, false, lines)
	vim.bo[buf].modifiable = false
	vim.bo[buf].buftype = "nofile"
	vim.keymap.set("n", "q", "<cmd>close<cr>", { buffer = buf, nowait = true })
end

-- ============================================================================
-- WORD SEARCH FUNCTIONALITY WITH NLP
-- ============================================================================

local function display_wordsearch()
	local text = get_buffer_text()
	if text:match("^%s*$") then
		vim.notify("Buffer is empty or contains only whitespace", vim.log.levels.WARN)
		return
	end

	if not check_dependencies() then
		return
	end

	local args = {
		"--grid-min",
		tostring(config.wordsearch.grid_size_min),
		"--grid-max",
		tostring(config.wordsearch.grid_size_max),
		"--max-words",
		tostring(config.wordsearch.max_words),
		"--semantic-groups",
		config.wordsearch.use_semantic_groups and "true" or "false",
	}

	local result = run_python_script("nlp_wordsearch.py", text, args)
	if not result then
		return
	end

	if not result.grid or not result.words then
		vim.notify("Could not generate word search from buffer content", vim.log.levels.WARN)
		return
	end

	-- Create display buffer
	local buf = vim.api.nvim_create_buf(false, true)
	local lines = {}

	-- Add grid
	for _, row in ipairs(result.grid) do
		table.insert(lines, table.concat(row, " "))
	end

	table.insert(lines, "")
	table.insert(lines, "Words to find (" .. #result.words .. "):")

	-- Group words by semantic categories if available
	if result.semantic_groups then
		for category, words in pairs(result.semantic_groups) do
			table.insert(lines, "")
			table.insert(lines, category:upper() .. ":")

			local words_per_line = 4
			for i = 1, #words, words_per_line do
				local word_line = ""
				for j = i, math.min(i + words_per_line - 1, #words) do
					word_line = word_line .. string.format("%-12s", words[j])
				end
				table.insert(lines, "  " .. word_line)
			end
		end
	else
		-- Regular word list
		local words_per_line = 4
		for i = 1, #result.words, words_per_line do
			local word_line = ""
			for j = i, math.min(i + words_per_line - 1, #result.words) do
				word_line = word_line .. string.format("%-12s", result.words[j])
			end
			table.insert(lines, word_line)
		end
	end

	if result.metadata and result.metadata.difficulty then
		table.insert(lines, "")
		table.insert(lines, "Difficulty: " .. result.metadata.difficulty)
	end

	table.insert(lines, "")
	table.insert(lines, "Press 'q' to close")

	local win = vim.api.nvim_open_win(buf, true, {
		relative = config.window.relative,
		width = math.max(60, #result.grid * 2 + 20),
		height = math.min(30, #lines + 2),
		col = 5,
		row = 2,
		border = config.window.border,
		title = "Enhanced Word Search",
	})

	vim.api.nvim_buf_set_lines(buf, 0, -1, false, lines)
	vim.bo[buf].modifiable = false
	vim.bo[buf].buftype = "nofile"
	vim.keymap.set("n", "q", "<cmd>close<cr>", { buffer = buf, nowait = true })
end

-- ============================================================================
-- PUBLIC API
-- ============================================================================

function M.find_anagrams()
	display_anagrams()
end

function M.generate_poetry(type)
	display_poetry(type)
end

function M.create_wordsearch()
	display_wordsearch()
end

function M.setup(opts)
	-- Merge user config with defaults
	if opts then
		config = vim.tbl_deep_extend("force", config, opts)
	end

	-- Create user commands
	vim.api.nvim_create_user_command("WordplayAnagrams", function()
		M.find_anagrams()
	end, { desc = "Find anagrams using NLP analysis" })

	vim.api.nvim_create_user_command("WordplayPoetry", function(args)
		M.generate_poetry(args.args ~= "" and args.args or nil)
	end, {
		nargs = "?",
		complete = function()
			return { "haiku", "limerick", "sonnet", "free_verse" }
		end,
		desc = "Generate poetry using NLP",
	})

	vim.api.nvim_create_user_command("WordplaySearch", function()
		M.create_wordsearch()
	end, { desc = "Create semantic word search" })

	vim.api.nvim_create_user_command("WordplayCheck", function()
		if check_dependencies() then
			vim.notify("All NLP dependencies are available!", vim.log.levels.INFO)
		end
	end, { desc = "Check NLP dependencies" })

	-- Set up keymaps if enabled
	if config.keymaps then
		if config.keymaps.anagrams then
			vim.keymap.set("n", config.keymaps.anagrams, M.find_anagrams, { desc = "Find anagrams (NLP)" })
		end
		if config.keymaps.poetry then
			vim.keymap.set("n", config.keymaps.poetry, function()
				M.generate_poetry()
			end, { desc = "Generate poetry (NLP)" })
		end
		if config.keymaps.wordsearch then
			vim.keymap.set("n", config.keymaps.wordsearch, M.create_wordsearch, { desc = "Create word search (NLP)" })
		end
	end
end

return M
