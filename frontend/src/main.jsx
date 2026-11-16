import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const format = (value, digits = 2) => value.toFixed(digits)
const frameUrl = (frame) => '/api/observations/' + frame.frame_id
const time = (frame) => new Date(frame.timestamp).toLocaleTimeString('zh-CN', { hour12: false })
const elapsed = (first, last) => Math.max(0, new Date(last.timestamp) - new Date(first.timestamp))
const duration = (ms) => {
  const seconds = Math.floor(ms / 1000)
  return Math.floor(seconds / 60) + ':' + String(seconds % 60).padStart(2, '0')
}

async function readJson(url, signal) {
  const response = await fetch(url, { signal })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || '观测读取失败')
  return result
}

function Coordinates({ label, point, unit, axes = ['X', 'Y', 'Z'] }) {
  return <div className="coordinates">
    <span>{label}</span>
    <dl>{point.map((value, index) => <div key={axes[index]}>
      <dt>{axes[index]}</dt><dd>{format(value)} <small>{unit}</small></dd>
    </div>)}</dl>
  </div>
}

function DetectionBox({ bbox, dimensions, confidence }) {
  if (!bbox) return null
  const [left, top, right, bottom] = bbox
  return <span className="detection" style={{
    left: 100 * left / dimensions.width + '%', top: 100 * top / dimensions.height + '%',
    width: 100 * (right - left) / dimensions.width + '%',
    height: 100 * (bottom - top) / dimensions.height + '%',
  }}>人员 {format(confidence * 100, 0)}%</span>
}

function SearchResults({ mission, frames, seek }) {
  if (!mission?.people_results) return null
  const people = mission.people_results
  const confirmed = people.filter((person) => person.status === 'confirmed').length
  const review = mission.events.findLast((event) => event.type === 'search_review')
  const expected = mission.alarm.expected_people
  return <section className="panel search-results" aria-label="搜救结果">
    <div className="panel-title"><h2>搜救结果</h2>
      <span>{mission.status === 'interrupted' ? '任务中断' : review?.source === 'planner' ? (review.provider === 'mock' ? '本地规则确认结束' : '大模型确认结束') : review?.source === 'search_limit' ? '搜索预算结束' : '历史任务'}</span>
    </div>
    <p>机上确认 {confirmed} / {expected} 人 · 待确认 {Math.max(0, expected - confirmed)} 人</p>
    {review?.rationale && <p className="review-detail">{review.provider} · {review.rationale}</p>}
    {review?.source === 'search_limit' && review.recommendation === 'continue' && <p className="review-detail">大模型建议继续，当前任务已用完搜索预算。</p>}
    {!people.length && <p>本次搜索没有生成可展示的人员估计位置。</p>}
    <div className="result-cards">
      {people.map((person) => {
        const index = frames.findIndex((frame) => frame.frame_id === person.evidence.frame_id)
        const frame = frames[index]
        return <article key={person.id} className="result-card">
          <div className="panel-title">
            <h3>人员候选 {person.id}</h3>
            <span>{person.status === 'confirmed' ? '多帧确认' : '单帧待复查'} · {person.observations} 帧</span>
          </div>
          {frame ? <button className="evidence-image" type="button" onClick={() => seek(index)}
            aria-label={`在回放中查看候选 ${person.id} 的原始画面`}>
            <img src={frameUrl(frame) + '/rgb'} alt={`人员候选 ${person.id} 的实采航拍图`} loading="lazy" />
            <DetectionBox {...person.evidence} dimensions={frame.camera_intrinsics} />
          </button> : <p>原始图像暂不可用</p>}
          <Coordinates label="AirSim NED" point={person.ned_m} axes={['北 X', '东 Y', '下 Z']} unit="m" />
          <Coordinates label="Unreal Engine" point={person.ue_cm} unit="cm" />
        </article>
      })}
    </div>
  </section>
}

function Track({ frames, step, point, basemap, events, alarm }) {
  const [zoom, setZoom] = useState(false)
  if (!basemap) return <p className="map-empty">此任务暂无已配准的 ESARBench 底图。</p>
  const width = basemap.width_px
  const height = basemap.height_px
  const locate = ([north, east]) => [
    width * (east - basemap.west) / (basemap.east - basemap.west),
    height * (basemap.north - north) / (basemap.north - basemap.south),
  ]
  const positions = frames.map((frame) => frame.drone_position)
  const [x, y] = locate(positions[step])
  const factor = zoom ? 3 : 1
  const w = width / factor
  const h = height / factor
  const left = Math.max(0, Math.min(width - w, x - w / 2))
  const top = Math.max(0, Math.min(height - h, y - h / 2))
  const marker = width / (97 * factor)
  const waypoints = events.filter((event) => event.waypoint_ned_m)
  const visibleFrames = new Set(frames.slice(0, step + 1).map((frame) => frame.frame_id))
  const detections = events.filter((event) => visibleFrames.has(event.frame_id))
    .flatMap((event) => event.detections || []).filter((detection) => detection.ned_m)
  const bar = w / 6
  const searchCorner = alarm && locate([alarm.north + alarm.half_width, alarm.east - alarm.half_width])
  const otherCorner = alarm && locate([alarm.north - alarm.half_width, alarm.east + alarm.half_width])
  return <div className="map-canvas">
    <svg viewBox={`${left} ${top} ${w} ${h}`} role="img" aria-label="ESARBench 罗布泊正射底图及 NED 航迹">
      <image href={basemap.image_url} width={width} height={height} />
      {alarm && <rect x={searchCorner[0]} y={searchCorner[1]} width={otherCorner[0] - searchCorner[0]}
        height={otherCorner[1] - searchCorner[1]} fill="#ffc77035" stroke="#ffd480" strokeWidth="2"
        strokeDasharray="7 5" vectorEffect="non-scaling-stroke"><title>模拟报警区域</title></rect>}
      {waypoints.map((event) => {
        const [wx, wy] = locate(event.waypoint_ned_m)
        const color = event.type === 'replan' ? '#ffd883' : '#fff'
        return <g key={event.timestamp}><circle cx={wx} cy={wy} r={marker * .8} stroke={color}
          fill="#503017" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          <text x={wx + marker * 1.4} y={wy} style={{ fontSize: marker * 1.7 }} className="map-label">
            {event.type === 'dispatch' ? '报警中心' : event.sector}
          </text></g>
      })}
      <polyline points={positions.slice(0, step + 1).map(locate).map((position) => position.join(',')).join(' ')}
        stroke="#66ffe5" strokeWidth="3" strokeLinejoin="round" vectorEffect="non-scaling-stroke" fill="none" />
      {detections.map((detection, index) => {
        const [dx, dy] = locate(detection.ned_m)
        return <circle key={index} cx={dx} cy={dy} r={marker} fill="#fb776b" stroke="white"
          strokeWidth="1.5" vectorEffect="non-scaling-stroke"><title>人员估计位置</title></circle>
      })}
      {point && <g>
        <line x1={x} y1={y} x2={locate(point)[0]} y2={locate(point)[1]} stroke="#ffe087"
          strokeDasharray="5 5" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        <circle cx={locate(point)[0]} cy={locate(point)[1]} r={marker} fill="#ffe087" stroke="#432f12"
          strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      </g>}
      <circle cx={x} cy={y} r={marker * 1.6} fill="#66ffe533" />
      <circle cx={x} cy={y} r={marker} fill="#5effde" stroke="#102521" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      <path d={`M ${left + w / 24} ${top + h * .94} h ${bar}`} stroke="#fff"
        strokeWidth="3" vectorEffect="non-scaling-stroke" />
      <text x={left + w / 24} y={top + h * .92} style={{ fontSize: marker * 1.7 }} className="map-label">
        {format((basemap.east - basemap.west) / factor / 6, 0)} m
      </text>
    </svg>
    <div className="map-tools"><span>↑ N</span><button type="button" onClick={() => setZoom(!zoom)}>
      {zoom ? '显示全图' : '放大航迹'}
    </button></div>
  </div>
}

function Replay({ frames, basemap, mission }) {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [selected, setSelected] = useState(null)
  const [projection, setProjection] = useState(null)
  const frame = frames[step]
  const events = mission?.events || []
  const current = events.find((event) => event.frame_id === frame.frame_id)
  const summary = events.find((event) => event.type === 'summary')
  const heights = frames.map((entry) => frames[0].drone_position[2] - entry.drone_position[2])
  const result = projection?.frame === frame.frame_id && projection?.pixel === selected ? projection : null

  useEffect(() => {
    if (!playing) return
    if (step === frames.length - 1) return
    const timer = setTimeout(() => {
      setStep(step + 1)
      if (step + 1 === frames.length - 1) setPlaying(false)
    }, elapsed(frames[step], frames[step + 1]) / speed)
    return () => clearTimeout(timer)
  }, [playing, step, speed, frames])

  useEffect(() => {
    if (!selected) return
    const controller = new AbortController()
    const query = new URLSearchParams(selected)
    readJson(frameUrl(frame) + '/point?' + query, controller.signal)
      .then((point) => setProjection({ frame: frame.frame_id, pixel: selected, point }))
      .catch((error) => {
        if (error.name !== 'AbortError') setProjection({ frame: frame.frame_id, pixel: selected, error: error.message })
      })
    return () => controller.abort()
  }, [frame, selected])

  function seek(index) {
    setPlaying(false)
    setStep(index)
    setSelected(null)
  }

  function pick(event) {
    const rect = event.currentTarget.getBoundingClientRect()
    const calibration = frame.depth_camera_intrinsics
    setPlaying(false)
    setSelected({
      u: Math.min(calibration.width - 1, Math.floor((event.clientX - rect.left) / rect.width * calibration.width)),
      v: Math.min(calibration.height - 1, Math.floor((event.clientY - rect.top) / rect.height * calibration.height)),
    })
  }

  const pixel = result?.point?.rgb_pixel
  return <>
    <div className="metrics">
      <div><span>场景</span><strong>{frame.map_id}</strong></div>
      <div><span>搜救任务</span><strong>#{frame.task_id}</strong></div>
      <div><span>观测帧</span><strong>{String(step + 1).padStart(2, '0')} <small>/ {frames.length}</small></strong></div>
      <div><span>较首帧高度变化</span><strong>{format(heights[step])} <small>m</small></strong></div>
    </div>
    <div className="workspace">
      <section className="panel map">
        <div className="panel-title"><h2>{{lopnur: '罗布泊', aotai2800: '鳌太线', k2: 'K2 雪山'}[frame.map_id] || frame.map_id} · 地形底图</h2><span>ESARBench 正射 · NED</span></div>
        <Track frames={frames} step={step} point={result?.point?.ned_m} basemap={basemap} events={events} alarm={mission?.alarm} />
        <div className="map-legend">
          <span><i className="track-dot" />无人机 / 航迹</span>
          <span><i className="point-dot" />重规划 / 像素定位</span>
          <span><i className="person-dot" />人员检测位置</span>
          {mission?.alarm && <span>▧ 模拟报警范围</span>}
        </div>
        <Coordinates label="无人机位置" point={frame.drone_position} axes={['北 X', '东 Y', '下 Z']} unit="m" />
      </section>
      <aside>
      <section className="panel camera">
        <div className="panel-title"><h2>前视相机</h2><span>{frame.camera_intrinsics.width} × {frame.camera_intrinsics.height} RGB</span></div>
        <div className="scene" onClick={pick}>
          <img src={frameUrl(frame) + '/rgb'} alt="ESARBench 无人机前视图像" draggable="false" />
          {pixel && <i className="crosshair" style={{
            left: 100 * (pixel[0] + 0.5) / frame.camera_intrinsics.width + '%',
            top: 100 * (pixel[1] + 0.5) / frame.camera_intrinsics.height + '%',
          }} />}
          {current?.detections?.map((detection, index) => <DetectionBox key={index} {...detection} dimensions={frame.camera_intrinsics} />)}
        </div>
        <div className="transport">
          <button type="button" disabled={frames.length < 2} onClick={() => {
            if (!playing) {
              if (step === frames.length - 1) setStep(0)
              setSelected(null)
            }
            setPlaying(!playing)
          }}>{playing ? 'Ⅱ 暂停' : '▶ 播放'}</button>
          <input type="range" aria-label="观测帧" min="0" max={frames.length - 1} value={step}
            disabled={frames.length < 2} onChange={(event) => seek(Number(event.target.value))} />
          <span title={time(frame)}>{duration(elapsed(frames[0], frame))} / {duration(elapsed(frames[0], frames.at(-1)))}</span>
          <select aria-label="回放速度" value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
            {[1, 2, 4].map((value) => <option key={value} value={value}>{value}×</option>)}
          </select>
        </div>
        <div className="help">点击图像查询地图坐标，也可以<button type="button" onClick={() => {
          setPlaying(false)
          setSelected({ u: 64, v: 68 })
        }}>查询示例像素</button>。</div>
      </section>
        {mission && <section className="panel mission">
          <div className="panel-title"><h2>协同记录</h2><span>{mission.followup_of ? '按候选点复查' : mission.planner} / {mission.detector}</span></div>
          <p>{mission.instruction}</p>
          {mission.alarm && <p>接警 · 约 {mission.alarm.expected_people} 人 · {mission.alarm.half_width * 2} 米见方</p>}
          {summary?.official_people != null && <p>复核匹配 {summary.verified_confirmed} / {summary.official_people} 人 · 含单帧候选 {summary.verified_candidates} 人</p>}
          {mission.offline_recheck && <p>航拍帧离线补检 · 复核匹配 {mission.offline_recheck.verified_confirmed} / {mission.offline_recheck.official_people} 人</p>}
          {summary?.horizontal_distance_m != null && <p>水平航迹 {format(summary.horizontal_distance_m, 0)} m · {summary.frames} 帧</p>}
          {current && <p>当前帧 · {current.detections.length} 个人员检测框 · 推理 {current.inference_ms} ms</p>}
          <ol>{events.filter((event) => !['summary', 'observation'].includes(event.type)).map((event) => <li key={event.timestamp}>
            {event.type === 'plan' && <><strong>初始规划 · {event.sector}</strong><span>{event.rationale}</span></>}
            {event.type === 'replan' && <><strong>观测后重规划 · {event.sector}</strong><span>{event.rationale}</span></>}
            {event.type === 'dispatch' && <strong>接警转场 · 前往报警区域</strong>}
            {event.type === 'search_review' && <strong>结束评估 · {event.action === 'finish' ? '结束搜索' : '继续搜索'}</strong>}
            {event.type === 'sector_complete' && <strong>{event.sector} · {mission.profile === 'dispatch' ? '累计候选位置' : '搜索结束 · 检测框'} {event.person_detections}</strong>}
            {event.type === 'error' && <strong>{event.error}</strong>}
          </li>)}</ol>
        </section>}
        <section className="panel location" aria-live="polite">
          <div className="panel-title"><h2>像素定位</h2><span>平面深度</span></div>
          {!selected && <p>选择画面中的像素，查看该点的 NED 和 UE 坐标。</p>}
          {selected && !result && <p>正在计算坐标…</p>}
          {result?.error && <p className="error">{result.error}</p>}
          {result?.point && <>
            <p>深度像素 ({selected.u}, {selected.v}) · {format(result.point.depth_m)} m</p>
            <Coordinates label="AirSim NED" point={result.point.ned_m} unit="m" />
            <Coordinates label="Unreal Engine" point={result.point.ue_cm} unit="cm" />
          </>}
        </section>
      </aside>
    </div>
    <SearchResults mission={mission} frames={frames} seek={(index) => {
      seek(index)
      document.querySelector('.camera')?.scrollIntoView({ block: 'nearest' })
    }} />
  </>
}

function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [selection, setSelection] = useState('')
  const frames = data?.frames
  const groups = new Map()
  const legacy = new Map()
  for (const frame of frames || []) {
    const task = frame.map_id + ' / #' + frame.task_id
    const previous = legacy.get(task)
    const timestamp = new Date(frame.timestamp).getTime()
    const group = frame.run_id || (previous && timestamp - previous.timestamp < 60_000 ? previous.group : frame.frame_id)
    if (!frame.run_id) legacy.set(task, { group, timestamp })
    if (!groups.has(group)) groups.set(group, [])
    groups.get(group).push(frame)
  }

  useEffect(() => {
    const controller = new AbortController()
    let revision = 0
    function refresh() {
      const requested = ++revision
      Promise.all(['/api/observations', '/api/basemaps', '/api/missions'].map((url) => readJson(url, controller.signal)))
        .then(([frames, maps, runs]) => {
          if (requested !== revision) return
          setError(null)
          setData({ frames, maps, runs })
        }).catch((reason) => {
          if (reason.name !== 'AbortError' && requested === revision) setError('无法加载观测，请确认 FastAPI 正在运行。')
        })
    }
    refresh()
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const updates = new WebSocket(protocol + '//' + window.location.host + '/api/updates')
    updates.onmessage = refresh
    return () => {
      controller.abort()
      updates.close()
    }
  }, [])
  const group = selection || [...groups.keys()].at(-1)
  const active = groups.get(group)?.[0]
  const basemap = data?.maps.find((map) => map.map_id === active?.map_id && map.task_id === active?.task_id)
  const mission = data?.runs.find((run) => run.run_id === active?.run_id)
  return <main>
    <header><div className="brand"><span className="mark" aria-hidden="true">⌖</span>Emergency UAV</div>
      <span className="tag">ESARBench · 离线回放</span></header>
    <div className="heading"><h1>无人机观测回放</h1>
      {group && <label>任务<select value={group} onChange={(event) => setSelection(event.target.value)}>
        {[...groups].map(([name, [frame]]) => {
          const run = data?.runs.find((entry) => entry.run_id === frame.run_id)
          const label = run?.followup_of ? '候选点复飞' : ({extended: '长航线', dispatch: '接警搜救'}[run?.profile] || '搜救')
          return <option key={name} value={name}>
            {frame.map_id} / #{frame.task_id} · {frame.run_id ? label + ' ' + frame.run_id.slice(0, 6) : '验证 ' + time(frame)}
          </option>
        })}
      </select></label>}</div>
    {error && <p role="alert">{error}</p>}
    {!frames && !error && <p>正在读取观测…</p>}
    {frames?.length === 0 && <p>暂无可回放的观测。请先运行 ESARBench 采集脚本。</p>}
    {active && <Replay key={group} frames={groups.get(group)} basemap={basemap} mission={mission} />}
  </main>
}

createRoot(document.getElementById('root')).render(<App />)
