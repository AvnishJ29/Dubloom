/** Thin API client: JSON requests, XHR uploads with progress, and SSE job subscriptions. */

/** An API failure with the server's message and HTTP status (0 when the request never completed). */
export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

function errorDetail(body, fallback) {
  if (typeof body?.detail === 'string') return body.detail
  if (Array.isArray(body?.detail))
    return body.detail
      .map((item) => `${item.loc?.slice(1).join('.') || 'Input'}: ${item.msg}`)
      .join('; ')
  return fallback
}

/** Fetch `/api` + `path` and return the parsed JSON body (null if none); throws ApiError when not ok. */
export async function request(path, options = {}) {
  const response = await fetch('/api' + path, { credentials: 'same-origin', ...options })
  let body = null
  try {
    body = await response.json()
  } catch {
    body = null
  }
  if (!response.ok) {
    const detail = errorDetail(body, 'The request could not be completed.')
    throw new ApiError(detail, response.status)
  }
  return body
}

/** POST `payload` as JSON; resolves like request(). */
export function post(path, payload) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  })
}

/** POST multipart `formData`, reporting the uploaded share (0..1) to `onProgress`; rejects with ApiError. */
export function upload(path, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api' + path)
    xhr.withCredentials = true
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total)
    }
    xhr.onerror = () =>
      reject(new ApiError('The upload failed. Check the connection and try again.', 0))
    xhr.onabort = () => reject(new ApiError('The upload was cancelled.', 0))
    xhr.onload = () => {
      let body = null
      try {
        body = JSON.parse(xhr.responseText)
      } catch {
        body = null
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body)
      else reject(new ApiError(errorDetail(body, 'The upload was rejected.'), xhr.status))
    }
    xhr.send(formData)
  })
}

/** Follow a job: `onData` gets every update, `onDone` runs once it is finished; returns a stop function.
 *  SSE accelerates updates; one polling loop also handles buffering proxies. */
export function subscribe(jobId, { onData, onDone, onError }) {
  let closed = false,
    source = null,
    timer = null,
    inFlight = false,
    misses = 0
  const stop = () => {
    closed = true
    clearTimeout(timer)
    timer = null
    source?.close()
  }
  const deliver = (data) => {
    if (closed) return
    onData(data)
    if (['completed', 'failed', 'cancelled'].includes(data.status)) {
      stop()
      onDone?.()
    }
  }
  const poll = async () => {
    if (closed || inFlight) return
    inFlight = true
    try {
      const data = await request('/jobs/' + jobId)
      if (closed) return
      misses = 0
      deliver(data)
    } catch (error) {
      if (closed) return
      misses += 1
      if (error.status === 401 || (error.status === 404 && misses >= 3)) {
        stop()
        onError?.(error)
        return
      }
      if (error.status !== 404) onError?.(error)
    } finally {
      inFlight = false
      if (!closed) timer = setTimeout(poll, 3000)
    }
  }
  if (typeof EventSource !== 'undefined') {
    source = new EventSource('/api/jobs/' + jobId + '/events', { withCredentials: true })
    source.onmessage = (event) => {
      let data
      try {
        data = JSON.parse(event.data)
      } catch {
        return
      }
      deliver(data)
    }
    source.addEventListener('done', () => {
      if (closed) return
      source.close()
      clearTimeout(timer)
      poll() // Read the final document even if the proxy omitted the last message.
    })
    source.addEventListener('gone', () => {
      if (closed) return
      stop()
      onError?.(new ApiError('The project no longer exists.', 404))
    })
  }
  poll()
  return stop
}
