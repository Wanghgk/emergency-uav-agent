import React, { useEffect, useState } from 'react'

export default function MissionLauncher({ busy, onStarted }) {
  const [open, setOpen] = useState(false)
  const [options, setOptions] = useState(null)
  const [map, setMap] = useState('')
  const [task, setTask] = useState('')
  const [scouts, setScouts] = useState(2)
  const [relays, setRelays] = useState(1)
  const [deepseek, setDeepseek] = useState(false)
  const [quick, setQuick] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    fetch('/api/mission-options', { signal: controller.signal }).then(async (response) => {
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || '无法载入地图')
      const first = result.tasks.find((item) => item.available)
      if (first) {
        setMap(first.map_id)
        setTask(String(first.task_id))
      }
      setOptions(result)
    }).catch((reason) => { if (reason.name !== 'AbortError') setError(reason.message) })
    return () => controller.abort()
  }, [open])

  function pickMap(value) {
    setMap(value)
    const first = options.tasks.find((item) => item.map_id === value && item.available)
    setTask(first ? String(first.task_id) : '')
  }

  async function start(event) {
    event.preventDefault()
    if (submitting || busy || !task) return
    setSubmitting(true)
    setError('')
    try {
      const response = await fetch('/api/missions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: Number(task), scouts, relays, deepseek, quick }),
      })
      const report = await response.json()
      if (!response.ok) throw new Error(typeof report.detail === 'string' ? report.detail : '任务参数无效。')
      onStarted(report)
      setOpen(false)
    } catch (reason) {
      setError(reason.message)
    } finally {
      setSubmitting(false)
    }
  }

  const tasks = options?.tasks || []
  const maps = [...new Map(tasks.map((item) => [item.map_id, item.map_name]))]
  const selected = tasks.find((item) => String(item.task_id) === task)
  return <section className="panel launcher" aria-label="启动新任务">
    <div className="panel-title">
      <h2>启动新任务</h2>
      <button type="button" aria-expanded={open} onClick={() => { setError(''); setOpen(!open) }}>
        {open ? '收起' : '选择地图和任务'}
      </button>
    </div>
    {open && <form onSubmit={start} className="launcher-form">
      {busy && <p role="status">已有机队任务，请在任务结束后启动新任务。</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {!options && !error && <p>正在读取可用地图…</p>}
      {options && <fieldset disabled={busy || submitting}>
        <div className="launcher-fields">
          <label>地图<select value={map} onChange={(event) => pickMap(event.target.value)}>
            <option value="" disabled>选择地图</option>
            {maps.map(([id, name]) => <option value={id} key={id}>{name}</option>)}
          </select></label>
          <label>任务<select value={task} onChange={(event) => setTask(event.target.value)}>
            <option value="" disabled>无可启动任务</option>
            {tasks.filter((item) => item.map_id === map).map((item) =>
              <option key={item.task_id} value={item.task_id} disabled={!item.available}>
                #{item.task_id}{item.available ? '' : ' · 地图未安装'}
              </option>)}
          </select></label>
          <label>感知机<select value={scouts} onChange={(event) => setScouts(Number(event.target.value))}>
            {[1, 2, 3, 4, 5, 6].map((value) => <option key={value}>{value}</option>)}
          </select></label>
          <label>通导机<select value={relays} onChange={(event) => setRelays(Number(event.target.value))}>
            {[0, 1, 2, 3, 4].map((value) => <option key={value}>{value}</option>)}
          </select></label>
        </div>
        {selected && <p>{selected.search_width_m} 米见方的报警估计范围 · {selected.instruction}</p>}
        <div className="launcher-actions">
          <label><input type="checkbox" checked={quick} onChange={(event) => setQuick(event.target.checked)} />快速试飞（每架 6 个航点）</label>
          <label><input type="checkbox" checked={deepseek} disabled={!options.deepseek_available}
            onChange={(event) => setDeepseek(event.target.checked)} />{options.deepseek_available ? '使用 DeepSeek' : 'DeepSeek 未配置'}</label>
          <button type="submit" disabled={!selected?.available}>{submitting ? '正在启动…' : '启动任务'}</button>
        </div>
      </fieldset>}
    </form>}
  </section>
}
