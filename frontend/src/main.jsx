import React, { useEffect, useId, useRef, useState } from 'react'
import { missionConclusion } from './replayTimeline.js'
import MissionLauncher from './MissionLauncher.jsx'
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
const missionStatuses = {
  starting: '正在启动', running: '进行中', completed: '已完成',
  finished: '已完成', partial: '部分完成', paused: '已暂停', failed: '执行失败', interrupted: '已中断',
}
function fleetDisplay(fleet) {
  const counts = { scout: 0, relay: 0 }
  const hues = { scout: [166, 208, 266, 130, 327, 190], relay: [42, 17, 70, 350] }
  return Object.fromEntries(Object.entries(fleet || {}).map(([name, role]) => {
    const index = counts[role]++
    return [name, { role, label: `${role === 'scout' ? '感知' : '通导'} ${index + 1}`,
      color: `hsl(${hues[role][index % hues[role].length]} 90% 72%)` }]
  }))
}

function radioLinks(sample) {
  return sample?.radio_links || Object.entries(sample?.links || {}).map(([to, link]) => (
    { ...link, from: 'Relay', to }
  ))
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
      <span>{mission.status === 'partial' ? '部分完成 · 有飞机执行失败' : mission.status === 'failed' ? '任务中断 · 部分结果' : mission.status === 'interrupted' ? '任务中断' : review?.source === 'planner' ? (review.provider === 'mock' ? '本地规则确认结束' : '大模型确认结束') : review?.source === 'search_limit' || mission.profile === 'fleet' ? '搜索预算结束' : '历史任务'}</span>
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

function Track({ frames, step, point, basemap, events, alarm, fleet, start, ground, groundRoute, latest = false }) {
  const [manualZoom, setZoom] = useState(null)
  const [pan, setPan] = useState(null)
  const [showPaths, setShowPaths] = useState(true)
  const [failedImageUrl, setFailedImageUrl] = useState(null)
  const searchClip = useId()
  const drag = useRef(null)
  if (!basemap) return <p className="map-empty">此任务暂无已配准的 ESARBench 底图。</p>
  const imageFailed = failedImageUrl === basemap.image_url
  const width = basemap.width_px
  const height = basemap.height_px
  const locate = ([north, east]) => [
    width * (east - basemap.west) / (basemap.east - basemap.west),
    height * (basemap.north - north) / (basemap.north - basemap.south),
  ]
  const positions = frames.map((frame) => frame.drone_position)
  const frame = frames[step]
  const telemetry = (fleet ? events : []).filter((event) => event.type === 'fleet_telemetry')
  const visibleTelemetry = latest || !frame ? telemetry :
    telemetry.filter((event) => Date.parse(event.timestamp) <= Date.parse(frame.timestamp))
  const snapshot = visibleTelemetry.at(-1)
  const visibleTime = latest || !frame ? null : frame.timestamp
  const visibleEvents = visibleTime ? events.filter((event) => Date.parse(event.timestamp) <= Date.parse(visibleTime)) : events
  const center = latest || !frame ? Object.values(snapshot?.positions || {})[0] || start : positions[step]
  const display = fleetDisplay(fleet)
  const [x, y] = locate(center)
  const base = events.find((event) => event.type === 'fleet_ready')?.base_ned_m || (fleet ? start : null)
  const basePixel = base && locate(base)
  const northRange = base && alarm ? [Math.min(base[0], alarm.north - alarm.half_width), Math.max(base[0], alarm.north + alarm.half_width)] : null
  const eastRange = base && alarm ? [Math.min(base[1], alarm.east - alarm.half_width), Math.max(base[1], alarm.east + alarm.half_width)] : null
  const defaultFocus = northRange ? locate([northRange[0] + (northRange[1] - northRange[0]) / 2,
    eastRange[0] + (eastRange[1] - eastRange[0]) / 2]) : [x, y]
  const fitZoom = northRange ? Math.max(1, Math.min(
    (basemap.north - basemap.south) / (northRange[1] - northRange[0] + 100),
    (basemap.east - basemap.west) / (eastRange[1] - eastRange[0] + 100)
  )) : 1
  const zoom = manualZoom ?? fitZoom
  const [focusX, focusY] = pan || defaultFocus
  const w = width / zoom
  const h = height / zoom
  const left = Math.max(0, Math.min(width - w, focusX - w / 2))
  const top = Math.max(0, Math.min(height - h, focusY - h / 2))
  const marker = width / (105 * zoom)
  const waypoints = (fleet ? visibleEvents : events).filter((event) => event.waypoint_ned_m)
  const visibleFrames = new Set(frames.slice(0, step + 1).map((frame) => frame.frame_id))
  const detections = events.filter((event) => visibleFrames.has(event.frame_id))
    .flatMap((event) => event.detections || []).filter((detection) => detection.ned_m)
  const bar = w / 6
  const searchCorner = alarm && locate([alarm.north + alarm.half_width, alarm.east - alarm.half_width])
  const otherCorner = alarm && locate([alarm.north - alarm.half_width, alarm.east + alarm.half_width])
  function adjustZoom(multiplier) {
    if (!pan) setPan(defaultFocus)
    setZoom(Math.max(1, Math.min(12, zoom * multiplier)))
  }

  return <div className="map-canvas">
    <svg className={zoom > 1 ? 'pannable-map' : undefined}
      viewBox={`${left} ${top} ${w} ${h}`} role="img" aria-label="ESARBench 正射底图及 NED 航迹"
      onPointerDown={(event) => {
        if (zoom <= 1 || event.button !== 0) return
        const rect = event.currentTarget.getBoundingClientRect()
        event.currentTarget.setPointerCapture(event.pointerId)
        drag.current = { px: event.clientX, py: event.clientY, cx: left + w / 2, cy: top + h / 2,
          scale: Math.max(w / rect.width, h / rect.height) }
      }}
      onPointerMove={(event) => {
        if (!drag.current) return
        const { px, py, cx, cy, scale } = drag.current
        setPan([Math.max(w / 2, Math.min(width - w / 2, cx - (event.clientX - px) * scale)),
          Math.max(h / 2, Math.min(height - h / 2, cy - (event.clientY - py) * scale))])
      }}
      onLostPointerCapture={() => { drag.current = null }}>
      <rect width={width} height={height} fill="#e3e8ec" />
      {!imageFailed && <image key={basemap.image_url} href={basemap.image_url} width={width} height={height}
        onError={() => setFailedImageUrl(basemap.image_url)} />}
      {alarm && <rect x={searchCorner[0]} y={searchCorner[1]} width={otherCorner[0] - searchCorner[0]}
        height={otherCorner[1] - searchCorner[1]} fill="#ffc77035" stroke="#ffd480" strokeWidth="2"
        strokeDasharray="7 5" vectorEffect="non-scaling-stroke"><title>报警估计范围</title></rect>}
      {alarm && <defs><clipPath id={searchClip}><rect x={searchCorner[0]} y={searchCorner[1]}
        width={otherCorner[0] - searchCorner[0]} height={otherCorner[1] - searchCorner[1]} /></clipPath></defs>}
      {showPaths && alarm && Object.entries(display).filter(([, aircraft]) => aircraft.role === 'scout').map(([name, aircraft]) => {
        const plan = visibleEvents.findLast((event) => event.type === 'fleet_plan' && event.vehicle_name === name)
        return plan && <polyline key={name} points={plan.waypoints_ned_xy.map(locate).map((xy) => xy.join(',')).join(' ')}
          clipPath={`url(#${searchClip})`} stroke={aircraft.color} strokeWidth="1.5" strokeDasharray="4 7"
          vectorEffect="non-scaling-stroke" fill="none">
          <title>{aircraft.label} 第 {plan.round} 轮计划航点</title>
        </polyline>
      })}
      {waypoints.map((event) => {
        const [wx, wy] = locate(event.waypoint_ned_m)
        const color = event.type === 'replan' ? '#ffd883' : '#fff'
        return <g key={`${event.timestamp}-${event.vehicle_name || event.type}`}><circle cx={wx} cy={wy} r={marker * .8} stroke={color}
          fill="#503017" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          <text x={wx + marker * 1.4} y={wy} style={{ fontSize: marker * 1.7 }} className="map-label">
            {event.type === 'dispatch' ? '报警大致位置' : event.sector}
          </text></g>
      })}
      {snapshot && basePixel && <text x={basePixel[0]} y={basePixel[1] + 3 * marker}
        className="map-label" style={{fontSize: marker * 1.7}}>基站</text>}
      {snapshot && Object.entries(display).map(([name, { color, label }]) => {
        if (!snapshot.positions[name]) return null
        const [fx, fy] = locate(snapshot.positions[name])
        return <g key={name}>
          <circle cx={fx} cy={fy} r={marker * .8} fill={color} stroke="#102521" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
          <text x={fx + marker} y={fy - marker} className="map-label" style={{fontSize: marker * 1.6}}>{label}</text>
        </g>
      })}
      {detections.map((detection, index) => {
        const [dx, dy] = locate(detection.ned_m)
        return <circle key={index} cx={dx} cy={dy} r={marker} fill="#fb776b" stroke="white"
          strokeWidth="1.5" vectorEffect="non-scaling-stroke"><title>人员估计位置</title></circle>
      })}
      {groundRoute?.waypoints_ned_m && <g>
        <polyline points={groundRoute.waypoints_ned_m.map(locate).map((xy) => xy.join(',')).join(' ')}
          stroke="#ff9bd7" strokeWidth="4" vectorEffect="non-scaling-stroke" fill="none">
          <title>地面救援拟定路线</title>
        </polyline>
        <circle cx={locate(groundRoute.waypoints_ned_m.at(-1))[0]} cy={locate(groundRoute.waypoints_ned_m.at(-1))[1]}
          r={marker * .66} fill="#ff9bd7"><title>地面路线接近点</title></circle>
      </g>}
      {groundRoute?.estimated_person_ned_m && <circle
        cx={locate(groundRoute.estimated_person_ned_m)[0]} cy={locate(groundRoute.estimated_person_ned_m)[1]}
        r={marker * .66} fill="#fb776b" stroke="white" strokeWidth="1.5" vectorEffect="non-scaling-stroke">
        <title>人员估计位置</title>
      </circle>}
      {ground?.entry && <g>
        <circle cx={locate(ground.entry.ned_m)[0]} cy={locate(ground.entry.ned_m)[1]} r={marker * .8}
          fill="#ff9bd7" stroke="#38203a" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        <text x={locate(ground.entry.ned_m)[0] + marker} y={locate(ground.entry.ned_m)[1] - marker}
          className="map-label" style={{ fontSize: marker * 1.6 }}>救援入口</text>
      </g>}
      {point && <g>
        <circle cx={locate(point)[0]} cy={locate(point)[1]} r={marker} fill="#ffe087" stroke="#432f12"
          strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      </g>}
      {!fleet && <><circle cx={x} cy={y} r={marker * 1.6} fill="#66ffe533" />
        <circle cx={x} cy={y} r={marker} fill="#5effde" stroke="#102521" strokeWidth="2" vectorEffect="non-scaling-stroke" /></>}
      <path d={`M ${left + w / 24} ${top + h * .94} h ${bar}`} stroke="#fff"
        strokeWidth="3" vectorEffect="non-scaling-stroke" />
      <text x={left + w / 24} y={top + h * .92} style={{ fontSize: marker * 1.7 }} className="map-label">
        {format((basemap.east - basemap.west) / zoom / 6, 0)} m
      </text>
    </svg>
    {imageFailed && <span className="map-notice">底图加载失败 · 显示坐标叠加</span>}
    {!imageFailed && basemap.render_source === 'depth_relief' &&
      <span className="map-notice">深度地形晕渲</span>}
    <div className="map-tools"><span>↑ N</span><div className="map-controls">
      <button type="button" aria-label="缩小地图" disabled={zoom === 1} onClick={() => adjustZoom(1 / 1.5)}>−</button>
      <button type="button" aria-label="放大地图" disabled={zoom >= 12} onClick={() => adjustZoom(1.5)}>＋</button>
      <button type="button" onClick={() => { setZoom(1); setPan(null) }}>全图</button>
      {fleet && alarm && <button type="button" aria-pressed={showPaths} onClick={() => setShowPaths(!showPaths)}>
        {showPaths ? '隐藏目标区路径' : '显示目标区路径'}
      </button>}
    </div></div>
  </div>
}

function FleetLinks({ sample, fleet }) {
  if (!sample) return <p>等待机队回传位置。</p>
  const display = fleetDisplay(fleet)
  const label = (name) => name === 'base' ? '基站' : display[name]?.label || name
  return <div className="map-legend">{radioLinks(sample).map((link) =>
    <span key={`${link.from}-${link.to}`}>{label(link.from)}—{label(link.to)} · {link.ok ? '连通' : '中断'} · {format(link.distance_m, 0)} m</span>
  )}</div>
}

function FleetStatus({ sample, fleet, live }) {
  if (!sample) return <p className="telemetry-waiting">等待机队回传位置。</p>
  const display = fleetDisplay(fleet)
  return <div className="telemetry">
    <div className="telemetry-heading"><strong>{live ? '机队实时位置' : '机队位置'}</strong><time dateTime={sample.timestamp}>采样于 {time(sample)}</time></div>
    <div className="telemetry-grid">{Object.entries(sample.positions).map(([name, position]) =>
      <div className="telemetry-vehicle" key={name}>
        <strong style={{ color: display[name]?.color }}>{display[name]?.label || name}</strong>
        <span title="北 / 东 / 下（米）">{position.map((value) => format(value, 1)).join(' / ')} m</span>
      </div>
    )}</div>
    <details><summary>通信链路状态</summary><FleetLinks sample={sample} fleet={fleet} /></details>
  </div>
}

function GroundRescue({ ground, route, selectRoute }) {
  if (!ground) return null
  const statuses = { surface_checked: ground.obstacle_clearance_m ? '通过坡度和障碍筛查' : '历史路线 · 仅复核过坡度',
    no_route: '没有合规接近路线', manual_review: '需要复核' }
  return <section className="panel mission" aria-label="地面救援路线">
    <div className="panel-title"><h2>地面救援路线</h2><span>ESARBench 表面坡度复核</span></div>
    {ground.entry && <p>入口 · {ground.entry.label}</p>}
    {ground.status === 'waiting_for_confirmation' && <p>等待多帧确认的人员位置。</p>}
    {ground.status === 'unavailable' && <p>地形规划不可用 · {ground.reason}</p>}
    {route && <>
      {ground.routes.length > 1 && <label>选择人员 <select value={route.person_id}
        onChange={(event) => selectRoute(Number(event.target.value))}>
        {ground.routes.map((item) => <option key={item.person_id} value={item.person_id}>人员 {item.person_id}</option>)}
      </select></label>}
      <p>{statuses[route.status]}{route.model && ` · ${route.model}`}</p>
      {route.metrics && <p>路线 {format(route.metrics.distance_m, 0)} m · 累计爬升 {format(route.metrics.ascent_m, 0)} m
        · 最大坡度 {format(route.metrics.max_slope_deg, 1)}° · 距估计人员 {format(route.metrics.approach_distance_m, 1)} m</p>}
      <p>{route.reason}</p>
    </>}
    {ground.status === 'reviewed' && <p>正射深度可能对应植被或雪面，尚需核验实际地面通行性。</p>}
  </section>
}

function CameraCard({ name, label, frame, detections, pixel, pick }) {
  return <section className="panel camera" id={`camera-${name}`} aria-label={label}>
    <div className="panel-title"><h2>{label}</h2><span>{frame ? time(frame) : '待回传'}</span></div>
    {frame ? <div className="scene" onClick={(event) => pick(event, frame)}>
      <img src={frameUrl(frame) + '/rgb'} alt={`${label}画面`} draggable="false" />
      {pixel && <i className="crosshair" style={{
        left: 100 * (pixel[0] + 0.5) / frame.camera_intrinsics.width + '%',
        top: 100 * (pixel[1] + 0.5) / frame.camera_intrinsics.height + '%',
      }} />}
      {detections?.map((detection, index) => <DetectionBox key={index} {...detection} dimensions={frame.camera_intrinsics} />)}
    </div> : <p className="camera-waiting">等待该机的首帧图像</p>}
  </section>
}

function FleetLive({ mission, basemap }) {
  const ground = mission.status !== 'failed' && missionConclusion(mission) ? mission.ground_rescue : null
  const failure = mission.events.findLast((event) => event.type === 'error')
  return <div className="workspace"><section className="panel map">
    <div className="panel-title"><h2>机队任务 · 转场与搜索</h2><span>ESARBench · NED</span></div>
    {mission.status === 'starting' && <p className="telemetry-waiting" role="status">正在启动 ESARBench 场景…</p>}
    {failure && <p className="telemetry-waiting" role="alert">{failure.error}</p>}
    <Track frames={[]} basemap={basemap} events={mission.events} alarm={mission.alarm}
      fleet={mission.fleet} start={mission.start_ned_m} ground={ground} latest />
    <FleetStatus fleet={mission.fleet} live={mission.status === 'running'}
      sample={mission.events.findLast((event) => event.type === 'fleet_telemetry')} />
    <GroundRescue ground={ground} />
  </section><div className="camera-grid">
    {Object.entries(fleetDisplay(mission.fleet)).filter(([, aircraft]) => aircraft.role === 'scout').map(([name, aircraft]) =>
      <CameraCard key={name} name={name} label={`${aircraft.label} · ${name}`} />)}
  </div></div>
}

function Replay({ frames, basemap, mission }) {
  const [cursor, setStep] = useState(0)
  const [follow, setFollow] = useState(mission?.profile === 'fleet' && mission?.status === 'running')
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [selected, setSelected] = useState(null)
  const [projection, setProjection] = useState(null)
  const [groundPerson, setGroundPerson] = useState(null)
  const events = mission?.events || []
  const conclusion = missionConclusion(mission, frames.at(-1))
  const lastStep = conclusion ? frames.length : frames.length - 1
  const step = follow ? lastStep : Math.min(cursor, lastStep)
  const atConclusion = Boolean(conclusion && step === lastStep)
  const frameStep = Math.min(step, frames.length - 1)
  const frame = frames[frameStep]
  const replayTime = atConclusion ? conclusion : frame
  const showCurrent = follow || atConclusion
  const current = events.find((event) => event.frame_id === frame.frame_id)
  const display = fleetDisplay(mission?.fleet)
  const cameras = mission?.fleet ? Object.entries(display).filter(([, aircraft]) => aircraft.role === 'scout') :
    [[frame.vehicle_name, { label: '前视相机' }]]
  const latestFrames = new Map(frames.slice(0, frameStep + 1).map((entry) => [entry.vehicle_name, entry]))
  const detections = new Map(events.filter((event) => event.type === 'observation').map((event) => [event.frame_id, event.detections]))
  const ground = atConclusion && mission.status !== 'failed' ? mission.ground_rescue : null
  const groundRoute = ground?.routes.find((item) => item.person_id === groundPerson) ||
    ground?.routes.find((item) => item.status === 'surface_checked') || ground?.routes[0]
  const summary = atConclusion ? events.find((event) => event.type === 'summary') : null
  const failure = mission?.status === 'failed' ? events.findLast((event) => event.type === 'error') : null
  const height = (mission?.fleet ? mission.start_ned_m : frames[0].drone_position)[2] - frame.drone_position[2]
  const result = projection?.pixel === selected ? projection : null
  const next = frames[step + 1] || (step < lastStep ? conclusion : null)
  const nextDelay = next ? elapsed(replayTime, next) / speed : null

  useEffect(() => {
    if (!playing || nextDelay === null) return
    const timer = setTimeout(() => {
      setStep(step + 1)
      if (step + 1 === lastStep) setPlaying(false)
    }, nextDelay)
    return () => clearTimeout(timer)
  }, [playing, step, nextDelay, lastStep])

  useEffect(() => {
    if (!selected) return
    const controller = new AbortController()
    const query = new URLSearchParams({ u: selected.u, v: selected.v })
    readJson(frameUrl(selected.frame) + '/point?' + query, controller.signal)
      .then((point) => setProjection({ pixel: selected, point }))
      .catch((error) => {
        if (error.name !== 'AbortError') setProjection({ pixel: selected, error: error.message })
      })
    return () => controller.abort()
  }, [selected])

  function seek(index) {
    setFollow(false)
    setPlaying(false)
    setStep(index)
    setSelected(null)
  }

  function selectPixel(cameraFrame, u, v) {
    setFollow(false)
    setStep(step)
    setPlaying(false)
    setSelected({ frame: cameraFrame, u, v })
  }

  function pick(event, cameraFrame) {
    const rect = event.currentTarget.getBoundingClientRect()
    const calibration = cameraFrame.depth_camera_intrinsics
    selectPixel(cameraFrame,
      Math.min(calibration.width - 1, Math.floor((event.clientX - rect.left) / rect.width * calibration.width)),
      Math.min(calibration.height - 1, Math.floor((event.clientY - rect.top) / rect.height * calibration.height)))
  }

  const pixel = result?.point?.rgb_pixel
  return <>
    <div className="metrics">
      <div><span>场景</span><strong>{frame.map_id}</strong></div>
      <div><span>搜救任务</span><strong>#{frame.task_id}</strong></div>
      <div><span>回放</span><strong>{atConclusion ? '任务结果' : <>{String(frameStep + 1).padStart(2, '0')} <small>/ {frames.length} 帧</small></>}</strong></div>
      <div><span>{mission?.fleet ? '较出发点高程' : '较首帧高度变化'}</span><strong>{format(height)} <small>m</small></strong></div>
    </div>
    <div className="workspace">
      <section className="panel map">
        <div className="panel-title"><h2>{{lopnur: '罗布泊', aotai2800: '鳌太线', k2: 'K2 雪山'}[frame.map_id] || frame.map_id} · 地形底图</h2><span>ESARBench 正射 · NED</span></div>
        <Track frames={frames} step={frameStep} point={result?.point?.ned_m} basemap={basemap} events={events}
          alarm={mission?.alarm} fleet={mission?.fleet} start={mission?.start_ned_m}
          ground={ground} groundRoute={groundRoute} latest={showCurrent} />
        <div className="map-legend">
          {mission?.fleet ? Object.entries(display).map(([name, aircraft]) =>
            <span key={name}><i style={{ background: aircraft.color }} />{aircraft.label} · {name}</span>)
            : <span><i className="track-dot" />无人机位置</span>}
          <span><i className="point-dot" />重规划 / 像素定位</span>
          <span><i className="person-dot" />人员检测位置</span>
          {groundRoute?.waypoints_ned_m && <span><i style={{ background: '#ff9bd7' }} />地面救援拟定路线</span>}
          {mission?.alarm && <span>▧ 报警估计范围</span>}
        </div>
        {mission?.fleet ? <FleetStatus fleet={mission.fleet} live={follow && mission.status === 'running'}
          sample={events.findLast((event) => event.type === 'fleet_telemetry' &&
          (showCurrent || Date.parse(event.timestamp) <= Date.parse(frame.timestamp)))} />
          : <Coordinates label="无人机位置" point={frame.drone_position} axes={['北 X', '东 Y', '下 Z']} unit="m" />}
      </section>
      <aside>
      <div className="camera-grid">
        {cameras.map(([name, aircraft]) => {
          const cameraFrame = latestFrames.get(name)
          return <CameraCard key={name} name={name} label={mission?.fleet ? `${aircraft.label} · ${name}` : aircraft.label}
            frame={cameraFrame} detections={detections.get(cameraFrame?.frame_id)}
            pixel={selected?.frame.frame_id === cameraFrame?.frame_id ? pixel : null} pick={pick} />
        })}
      </div>
      <section className="panel">
        <div className="transport">
          {mission?.fleet && mission.status === 'running' && <button type="button" aria-pressed={follow} onClick={() => {
            setPlaying(false)
            setStep(step)
            setSelected(null)
            setFollow(!follow)
          }}>{follow ? '跟随实时' : '回到实时'}</button>}
          <button type="button" disabled={lastStep === 0} onClick={() => {
            setFollow(false)
            if (!playing) {
              setStep(step === lastStep ? 0 : step)
              setSelected(null)
            }
            setPlaying(!playing)
          }}>{playing ? 'Ⅱ 暂停' : '▶ 播放'}</button>
          <input type="range" aria-label="任务回放进度" min="0" max={lastStep} value={step}
            disabled={lastStep === 0} onChange={(event) => seek(Number(event.target.value))} />
          <span title={time(replayTime)}>{duration(elapsed(frames[0], replayTime))} / {duration(elapsed(frames[0], conclusion || frames.at(-1)))}</span>
          {conclusion && <button type="button" onClick={() => seek(lastStep)}>查看结果</button>}
          <select aria-label="回放速度" value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
            {[1, 2, 4].map((value) => <option key={value} value={value}>{value}×</option>)}
          </select>
        </div>
        <div className="help">点击任意相机画面查询该帧的地图坐标，也可以<button type="button" onClick={() =>
          selectPixel(frame, Math.min(64, frame.depth_camera_intrinsics.width - 1),
            Math.min(68, frame.depth_camera_intrinsics.height - 1))
        }>查询示例像素</button>。</div>
      </section>
        {mission && <section className="panel mission">
          <div className="panel-title"><h2>协同记录</h2><span>{mission.followup_of ? '按候选点复查' : mission.planner} / {mission.detector}</span></div>
          {failure && <p role="alert">{failure.error}</p>}
          <p>{mission.instruction}</p>
          {mission.alarm && <p>接警 · 约 {mission.alarm.expected_people} 人 · 估计范围 {mission.alarm.half_width * 2} 米见方</p>}
          {summary?.official_people != null && <p>复核匹配 {summary.verified_confirmed} / {summary.official_people} 人 · 含单帧候选 {summary.verified_candidates} 人</p>}
          {atConclusion && mission.offline_recheck && <p>航拍帧离线补检 · 复核匹配 {mission.offline_recheck.verified_confirmed} / {mission.offline_recheck.official_people} 人</p>}
          {summary?.horizontal_distance_m != null && <p>水平航迹 {format(summary.horizontal_distance_m, 0)} m · {summary.frames} 帧</p>}
          {current && <p>当前帧 · {current.detections.length} 个人员检测框 · 推理 {current.inference_ms} ms</p>}
          <ol className="mission-scroll" tabIndex={0} aria-label="协同日志，最新事件在前">{events.flatMap((event, index) =>
            ['summary', 'observation', 'fleet_telemetry', 'fleet_feedback'].includes(event.type) ||
            (!showCurrent && Date.parse(event.timestamp) > Date.parse(frame.timestamp)) ||
            (!atConclusion && event.type === 'ground_rescue') ? [] : [[event, index]]
          ).reverse().map(([event, index]) => <li key={index}>
            <time dateTime={event.timestamp}>{time(event)}</time>
            {event.type === 'plan' && <><strong>初始规划 · {event.sector}</strong><span>{event.rationale}</span></>}
            {event.type === 'replan' && <><strong>观测后重规划 · {event.sector}</strong><span>{event.rationale}</span></>}
            {event.type === 'dispatch' && <strong>接警转场 · 前往报警区域</strong>}
            {event.type === 'search_review' && <strong>结束评估 · {event.action === 'finish' ? '结束搜索' : '继续搜索'}</strong>}
            {event.type === 'fleet_ready' && <strong>{cameras.length} 架感知机与 {Object.values(display).filter((item) => item.role === 'relay').length} 架通导机就绪</strong>}
            {event.type === 'fleet_plan' && <><strong>{display[event.vehicle_name]?.label} · 第 {event.round} 轮 · {event.sector}</strong><span>{event.rationale}</span></>}
            {event.type === 'fleet_relay_plan' && <><strong>{display[event.vehicle_name]?.label || '通导'}位置复评 · {event.accepted ? '候选通过' : '候选已更新'}</strong><span>{event.rationale}</span></>}
            {event.type === 'fleet_relay_move' && <strong>{display[event.vehicle_name]?.label || '通导'}改点 · {event.sector} · {event.source}</strong>}
            {event.type === 'fleet_end' && <strong>任务结束 · {event.unforwarded_frames} 帧尚未转发</strong>}
            {event.type === 'paused' && <strong>用户暂停任务</strong>}
            {event.type === 'fleet_capture_error' && <strong>{display[event.vehicle_name]?.label} · 图像采集失败，跳过该帧</strong>}
            {event.type === 'fleet_round_complete' && <strong>{display[event.vehicle_name]?.label} · 第 {event.round} 轮搜索完成</strong>}
            {event.type === 'fleet_aircraft_error' && <><strong>{display[event.vehicle_name]?.label} · 执行失败</strong><span>{event.error}</span></>}
            {event.type === 'ground_rescue' && <strong>地面救援规划 · {event.status === 'waiting_for_confirmation' ? '等待确认' : '已记录'}</strong>}
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
            <p>{mission?.fleet && `${display[selected.frame.vehicle_name]?.label} · `}深度像素 ({selected.u}, {selected.v}) · {format(result.point.depth_m)} m</p>
            <Coordinates label="AirSim NED" point={result.point.ned_m} unit="m" />
            <Coordinates label="Unreal Engine" point={result.point.ue_cm} unit="cm" />
          </>}
        </section>
        <GroundRescue ground={ground} route={groundRoute} selectRoute={setGroundPerson} />
      </aside>
    </div>
    {atConclusion && <SearchResults mission={mission} frames={frames} seek={(index) => {
      seek(index)
      document.getElementById(`camera-${frames[index].vehicle_name}`)?.scrollIntoView({ block: 'nearest' })
    }} />}
  </>
}

function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [selection, setSelection] = useState('')
  const [confirmRemoval, setConfirmRemoval] = useState(null)
  const [pendingAction, setPendingAction] = useState(null)
  const [actionError, setActionError] = useState('')
  const refreshMissions = useRef(null)
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
  for (const run of data?.runs || []) {
    if (!groups.has(run.run_id)) groups.set(run.run_id, [])
  }

  useEffect(() => {
    const controller = new AbortController()
    let loading = false
    let queued = false
    function refresh() {
      if (loading) {
        queued = true
        return
      }
      loading = true
      Promise.all(['/api/observations', '/api/basemaps', '/api/missions'].map((url) => readJson(url, controller.signal)))
        .then(([frames, maps, runs]) => {
          setError(null)
          setData({ frames, maps, runs })
        }).catch((reason) => {
          if (reason.name !== 'AbortError') setError('无法加载观测，请确认 FastAPI 正在运行。')
        }).finally(() => {
          loading = false
          if (queued && !controller.signal.aborted) {
            queued = false
            refresh()
          }
        })
    }
    refreshMissions.current = refresh
    refresh()
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const updates = new WebSocket(protocol + '//' + window.location.host + '/api/updates')
    updates.onmessage = refresh
    return () => {
      controller.abort()
      updates.close()
      refreshMissions.current = null
    }
  }, [])
  const running = data?.runs.findLast((run) => run.can_pause)
  const latestRecordedGroup = [...groups].findLast(([, entries]) => entries.length)
  const group = (groups.has(selection) ? selection : null) || running?.run_id || latestRecordedGroup?.[0] || [...groups.keys()].at(-1)
  const active = groups.get(group)?.[0]
  const mission = data?.runs.find((run) => run.run_id === group)
  const basemap = data?.maps.find((map) => map.map_id === (active?.map_id || mission?.task.map_name) &&
    map.task_id === (active?.task_id ?? mission?.task.task_id))
  const canDelete = Boolean(mission)

  async function pauseMission(runId) {
    setSelection(runId)
    setPendingAction('pause')
    setActionError('')
    try {
      const response = await fetch('/api/missions/' + encodeURIComponent(runId) + '/pause', { method: 'POST' })
      const report = await response.json()
      if (!response.ok) throw new Error(report.detail || '无法暂停任务。')
      setData((current) => current && ({
        ...current,
        runs: current.runs.map((run) => run.run_id === runId ? report : run),
      }))
    } catch (reason) {
      setActionError(reason.message)
    } finally {
      refreshMissions.current?.()
      setPendingAction(null)
    }
  }

  async function removeMission(runId) {
    setPendingAction('delete')
    setActionError('')
    try {
      const response = await fetch('/api/missions/' + encodeURIComponent(runId), { method: 'DELETE' })
      if (!response.ok) {
        const detail = await response.json()
        throw new Error(detail.detail || '无法删除任务。')
      }
      setData((current) => current && ({
        ...current,
        runs: current.runs.filter((run) => run.run_id !== runId),
        frames: current.frames.filter((frame) => frame.run_id !== runId),
      }))
      setSelection((value) => value === runId ? '' : value)
      setConfirmRemoval(null)
    } catch (reason) {
      setActionError(reason.message)
    } finally {
      refreshMissions.current?.()
      setPendingAction(null)
    }
  }
  return <main>
    <header><div className="brand"><span className="mark" aria-hidden="true">⌖</span>Emergency UAV</div>
      <span className="tag">ESARBench · 任务监控与回放</span></header>
    <div className="heading"><h1>无人机观测回放</h1>
      {group && <div className="mission-picker"><label>任务<select value={group} disabled={Boolean(pendingAction)} onChange={(event) => {
        setSelection(event.target.value)
        setConfirmRemoval(null)
        setActionError('')
      }}>
        {[...groups].map(([name, [frame]]) => {
          const run = data?.runs.find((entry) => entry.run_id === name)
          const label = run?.followup_of ? '候选点复飞' : ({extended: '长航线', dispatch: '接警搜救', fleet: '机队协同'}[run?.profile] || '搜救')
          return <option key={name} value={name}>
            {run && missionStatuses[run.status] ? `[${missionStatuses[run.status]}] ` : ''}
            {frame?.map_id || run.task.map_name} / #{frame?.task_id ?? run.task.task_id} · {run || frame.run_id ? label + ' ' + name.slice(0, 6) : '验证 ' + time(frame)}
          </option>
        })}
      </select></label>
      {missionStatuses[mission?.status] && <span className={`mission-badge mission-${mission.status}`} role="status">
        {missionStatuses[mission.status]}
      </span>}
      {mission?.can_pause && <button className="pause-mission" type="button" disabled={Boolean(pendingAction)}
        title="停止当前仿真并保留已采集的记录" onClick={() => pauseMission(mission.run_id)}>
        {pendingAction === 'pause' ? '正在暂停…' : '暂停任务'}
      </button>}
      {canDelete && <button className="delete-mission" type="button" disabled={Boolean(pendingAction)}
        onClick={() => {
          setSelection(mission.run_id)
          setConfirmRemoval(mission.run_id)
        }}>删除任务</button>}
      </div>}</div>
    {canDelete && confirmRemoval === mission.run_id && <div className="panel mission-remove">
      <span>删除 {mission.run_id.slice(0, 6)}？{mission.can_pause && '将先停止仿真。'}任务、观测和相关结果将永久删除。</span>
      <button type="button" disabled={Boolean(pendingAction)} onClick={() => setConfirmRemoval(null)}>取消</button>
      <button className="delete-mission" type="button" disabled={Boolean(pendingAction)}
        onClick={() => removeMission(mission.run_id)}>{pendingAction === 'delete' ? '正在删除…' : '确认删除'}</button>
    </div>}
    {actionError && <p role="alert">{actionError}</p>}
    <MissionLauncher busy={Boolean(running)} onStarted={(report) => {
      setData((current) => {
        const previous = current || { frames: [], maps: [], runs: [] }
        return previous.runs.some((run) => run.run_id === report.run_id) ? previous :
          { ...previous, runs: [...previous.runs, report] }
      })
      setSelection(report.run_id)
      refreshMissions.current?.()
    }} />
    {error && <p role="alert">{error}</p>}
    {!frames && !error && <p>正在读取观测…</p>}
    {frames?.length === 0 && !mission && <p>暂无可回放的观测。可以先启动一个新任务。</p>}
    {active && <Replay key={group} frames={groups.get(group)} basemap={basemap} mission={mission} />}
    {!active && mission?.fleet && <FleetLive mission={mission} basemap={basemap} />}
  </main>
}

createRoot(document.getElementById('root')).render(<App />)
