// Debug: Log when script loads
console.log("🔹 File manager script loaded");

// Helper function to format file size
function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// Helper function to format date
function formatDate(timestamp) {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString();
  } catch (e) {
    return "N/A";
  }
}

// Helper function to get file icon
function getFileIcon(filename) {
  if (!filename) return "file";
  const ext = filename.split(".").pop().toLowerCase();
  const iconMap = {
    pdf: "file-pdf",
    doc: "file-word",
    docx: "file-word",
    xls: "file-excel",
    xlsx: "file-excel",
    ppt: "file-powerpoint",
    pptx: "file-powerpoint",
    txt: "file-alt",
    text: "file-alt",
    json: "file-code",
    js: "file-code",
    py: "file-code",
    jpg: "file-image",
    jpeg: "file-image",
    png: "file-image",
    gif: "file-image",
    webp: "file-image",
    svg: "file-image",
    zip: "file-archive",
    rar: "file-archive",
    "7z": "file-archive",
    mp3: "file-audio",
    wav: "file-audio",
    ogg: "file-audio",
    mp4: "file-video",
    avi: "file-video",
    mov: "file-video",
    csv: "file-csv",
  };
  return iconMap[ext] || "file";
}

// Load and display the list of uploaded files
async function loadFileList() {
  console.log("🔹 loadFileList() called");
  const fileListContainer = document.getElementById("file-list");

  if (!fileListContainer) {
    console.error("❌ Không tìm thấy container cho danh sách tệp");
    return;
  }

  try {
    console.log("🔄 Đang tải danh sách tệp...");
    fileListContainer.innerHTML = `
      <div class="text-center py-4">
        <div class="inline-flex items-center text-slate-400">
          <i class="fas fa-spinner fa-spin mr-2"></i>
          <span>Đang tải danh sách tệp...</span>
        </div>
      </div>`;

    const timestamp = new Date().getTime();
    const apiUrl = `/api/files?_=${timestamp}`;
    console.log("🌐 Gửi yêu cầu đến:", apiUrl);

    const response = await fetch(apiUrl, {
      method: "GET",
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        Pragma: "no-cache",
        Expires: "0",
      },
      credentials: "same-origin",
    });

    console.log("✅ Phản hồi từ API:", {
      status: response.status,
      statusText: response.statusText,
      ok: response.ok,
      url: response.url,
    });

    if (!response.ok) {
      throw new Error(`Lỗi máy chủ: ${response.status} ${response.statusText}`);
    }

    const files = await response.json();
    console.log("📂 Danh sách tệp nhận được:", files);

    if (!files || !Array.isArray(files)) {
      throw new Error("Định dạng dữ liệu không hợp lệ từ máy chủ");
    }

    if (files.length === 0) {
      fileListContainer.innerHTML = `
        <div class="text-center py-4">
          <div class="text-slate-400 text-sm">
            <i class="far fa-folder-open mr-1"></i>
            <span>Chưa có tệp nào được tải lên</span>
          </div>
        </div>`;
      return;
    }

    // Generate file list HTML
    const filesHtml = files
      .map((file) => {
        const fileName = file.name;
        const fileSize = file.size ? formatFileSize(file.size) : "N/A";
        const uploadedDate = formatDate(file.uploaded || file.mtime);
        const fileIcon = getFileIcon(file.name);
        const fileUrl = `/uploads/${encodeURIComponent(file.name)}`;

        return `
        <div class="group relative flex items-center justify-between bg-slate-800/50 hover:bg-slate-700/50 p-3 rounded-lg border border-slate-700/50 transition-colors mb-2">
          <div class="flex items-center space-x-3 flex-1 min-w-0">
            <div class="relative flex-shrink-0">
              <i class="fas fa-${fileIcon} text-indigo-400 text-lg"></i>
            </div>
            <div class="min-w-0 flex-1">
              <p class="text-sm font-medium text-white truncate" title="${fileName}">${fileName}</p>
              <p class="text-xs text-slate-400">${fileSize} • ${uploadedDate}</p>
            </div>
          </div>
          <div class="flex space-x-2">
            <button data-filename="${fileName}" class="process-file-btn text-slate-400 hover:text-green-400 p-1 rounded-full hover:bg-slate-700/50 transition-colors" title="Xử lý file">
              <i class="fas fa-cogs"></i>
            </button>
            <a href="${fileUrl}" download="${fileName}" class="text-slate-400 hover:text-indigo-400 p-1 rounded-full hover:bg-slate-700/50 transition-colors" title="Tải xuống">
              <i class="fas fa-download"></i>
            </a>
            <button data-filename="${fileName}" class="delete-file-btn text-slate-400 hover:text-red-400 p-1 rounded-full hover:bg-slate-700/50 transition-colors" title="Xóa">
              <i class="fas fa-trash"></i>
            </button>
          </div>
        </div>`;
      })
      .join("");

    // Update the file list container
    fileListContainer.innerHTML = `
      <div class="space-y-2">
        ${filesHtml}
      </div>`;

    // Add click handlers for delete buttons
    document.querySelectorAll(".delete-file-btn").forEach((button) => {
      button.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        const filename = this.getAttribute("data-filename");
        if (confirm(`Bạn có chắc chắn muốn xóa ${filename}?`)) {
          deleteFile(filename);
        }
      });
    });
  } catch (error) {
    console.error("❌ Lỗi khi tải danh sách tệp:", error);
    fileListContainer.innerHTML = `
      <div class="p-4 text-center text-red-400">
        <i class="fas fa-exclamation-triangle mr-2"></i>
        <span>Không thể tải danh sách tệp: ${error.message}</span>
        <button onclick="loadFileList()" class="ml-2 px-2 py-1 bg-slate-700/50 hover:bg-slate-600/50 rounded text-sm">
          <i class="fas fa-sync-alt mr-1"></i> Thử lại
        </button>
      </div>`;
  }
}

// Delete a file
async function deleteFile(filename) {
  if (!confirm(`Bạn có chắc chắn muốn xóa ${filename}?`)) {
    return;
  }

  try {
    const response = await fetch(`/api/files/${encodeURIComponent(filename)}`, {
      method: "DELETE",
    });

    if (!response.ok) {
      throw new Error(`Lỗi ${response.status}: ${response.statusText}`);
    }

    // Reload the file list after successful deletion
    loadFileList();
  } catch (error) {
    console.error("Lỗi khi xóa tệp:", error);
    alert(`Lỗi khi xóa tệp: ${error.message}`);
  }
}

// Process a file and show analysis results
async function processFile(filename) {
  let fileElement;
  let originalHtml;

  try {
    console.log(`🔄 Đang xử lý file: ${filename}`);

    // Show loading state
    fileElement = document
      .querySelector(`button[data-filename="${filename}"]`)
      .closest(".group");
    originalHtml = fileElement.innerHTML;
    fileElement.innerHTML = `
      <div class="flex items-center justify-center w-full py-2">
        <i class="fas fa-spinner fa-spin text-indigo-400 mr-2"></i>
        <span class="text-sm text-slate-300">Đang xử lý...</span>
      </div>`;

    // Call the API to process the file
    const response = await fetch(
      `/api/files/process/${encodeURIComponent(filename)}?analyze=true`
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Lỗi khi xử lý file");
    }

    const data = await response.json();

    if (data.success) {
      // Show the analysis results in a modal
      showAnalysisModal(data);
    } else {
      throw new Error(data.error || "Không thể xử lý file");
    }
  } catch (error) {
    console.error("❌ Lỗi khi xử lý file:", error);
    alert(`Lỗi: ${error.message}`);
  } finally {
    // Restore the original content
    if (fileElement && originalHtml) {
      fileElement.innerHTML = originalHtml;
    }
  }
}

// Show analysis results in a modal
function showAnalysisModal(data) {
  // Create modal HTML
  const modalId = "analysis-modal";
  let modal = document.getElementById(modalId);

  // If modal doesn't exist, create it
  if (!modal) {
    modal = document.createElement("div");
    modal.id = modalId;
    modal.className = "analysis-panel p-4";
    modal.innerHTML = `
      <div class="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] flex flex-col">
        <div class="flex items-center justify-between p-4 border-b border-gray-200">
          <h3 class="text-lg font-semibold text-gray-900">Kết quả phân tích</h3>
          <button class="text-gray-500 hover:text-gray-700" onclick="document.getElementById('${modalId}').classList.add('hidden')">
            <i class="fas fa-times"></i>
          </button>
        </div>
        <div class="p-4 overflow-y-auto flex-1">
          <!-- Content will be inserted here -->
        </div>
        <div class="p-4 border-t border-gray-200 flex justify-end">
          <button class="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-md" 
                  onclick="document.getElementById('${modalId}').classList.add('hidden')">
            Đóng
          </button>
        </div>
      </div>`;
    document.body.appendChild(modal);
  }

  // Prepare content
  const content = `
    <div class="space-y-6">
      <!-- File Info -->
      <div class="bg-gray-50 p-4 rounded-lg">
        <h4 class="text-sm font-medium text-gray-900 mb-2">Thông tin file</h4>
        <div class="grid grid-cols-2 gap-2 text-sm">
          <div class="text-gray-600">Tên file:</div>
          <div class="text-gray-900 font-mono">${data.filename}</div>
          <div class="text-slate-400">Kích thước:</div>
          <div class="text-white">${formatFileSize(data.file_size)}</div>
          <div class="text-slate-400">Định dạng:</div>
          <div class="text-white">${data.file_type
            .toUpperCase()
            .replace(".", "")}</div>
        </div>
      </div>
      
      <!-- Analysis Results -->
      ${
        data.analysis
          ? `
      <div class="space-y-4">
        <!-- Summary -->
        <div>
          <h4 class="text-sm font-medium text-slate-300 mb-2">Tóm tắt</h4>
          <div class="bg-slate-700/30 p-4 rounded-lg text-slate-200 text-sm leading-relaxed">
            ${data.analysis.summary || "Không có tóm tắt"}
          </div>
        </div>
        
        <!-- Keywords -->
        <div>
          <h4 class="text-sm font-medium text-slate-300 mb-2">Từ khóa chính</h4>
          <div class="flex flex-wrap gap-2">
            ${
              data.analysis.keywords
                ? data.analysis.keywords
                    .map(
                      (kw) =>
                        `<span class="px-2 py-1 bg-indigo-600/30 text-indigo-300 text-xs rounded-full">
                ${kw.word} <span class="text-indigo-400 text-xs">(${kw.count})</span>
              </span>`
                    )
                    .join("")
                : "Không có từ khóa"
            }
          </div>
        </div>
        
        <!-- Category -->
        <div>
          <h4 class="text-sm font-medium text-slate-300 mb-2">Phân loại</h4>
          <div class="inline-block px-3 py-1 bg-slate-700/50 text-amber-400 text-sm rounded-full">
            ${data.analysis.category || "Chưa xác định"}
          </div>
        </div>
      </div>
      `
          : `
      <div class="text-center py-8 text-slate-400">
        <i class="fas fa-info-circle text-2xl mb-2"></i>
        <p>Không có thông tin phân tích</p>
      </div>
      `
      }
      
      <!-- Content Preview -->
      <div>
        <h4 class="text-sm font-medium text-slate-300 mb-2">Nội dung</h4>
        <div class="bg-slate-900/50 p-4 rounded-lg max-h-60 overflow-y-auto text-sm text-slate-300 font-mono whitespace-pre-wrap">
          ${
            data.content.length > 1000
              ? data.content.substring(0, 1000) + "..."
              : data.content
          }
        </div>
        ${
          data.content.length > 1000
            ? `
        <div class="text-xs text-slate-500 mt-1">
          Đang hiển thị 1000/${data.content.length} ký tự đầu tiên
        </div>
        `
            : ""
        }
      </div>
    </div>`;

  // Update modal content and show it
  modal.querySelector(".overflow-y-auto").innerHTML = content;
  modal.classList.remove("hidden");
}

// Show file content in a modal
function showFileContentModal(filename, content) {
  const modalHTML = `
    <div id="fileContentModal" class="file-content-panel p-4">
      <div class="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[80vh] flex flex-col mx-auto">
        <div class="flex justify-between items-center p-4 border-b border-gray-200">
          <h3 class="text-lg font-medium text-gray-900">Nội dung file: ${filename}</h3>
          <button onclick="document.getElementById('fileContentModal').remove()" class="text-gray-500 hover:text-gray-700">
            <i class="fas fa-times"></i>
          </button>
        </div>
        <div class="p-4 overflow-y-auto flex-1">
          <pre class="whitespace-pre-wrap text-sm text-gray-800 bg-gray-50 p-4 rounded">${content}</pre>
        </div>
        <div class="p-4 border-t border-gray-200 flex justify-end">
          <button onclick="document.getElementById('fileContentModal').remove()" class="px-4 py-2 bg-gray-100 text-gray-800 rounded hover:bg-gray-200 transition-colors">
            Đóng
          </button>
        </div>
      </div>
    </div>`;

  // Remove existing modal if any
  const existingModal = document.getElementById("fileContentModal");
  if (existingModal) {
    existingModal.remove();
  }

  // Add the new modal
  document.body.insertAdjacentHTML("beforeend", modalHTML);
}

// Initialize when the DOM is fully loaded
document.addEventListener("DOMContentLoaded", function () {
  console.log("🏁 DOM fully loaded");

  // Load files after a short delay to ensure everything is ready
  setTimeout(() => {
    loadFileList();
  }, 100);

  // Add click handler for process file buttons
  document.addEventListener("click", function (e) {
    if (e.target.closest(".process-file-btn")) {
      const button = e.target.closest(".process-file-btn");
      const filename = button.getAttribute("data-filename");
      processFile(filename);
    }
  });
});
