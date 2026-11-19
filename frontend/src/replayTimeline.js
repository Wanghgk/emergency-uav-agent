const completed = new Set(['completed', 'finished', 'partial', 'interrupted', 'failed'])

export function missionConclusion(mission, lastFrame) {
  if (!completed.has(mission?.status) || !(mission.people_results || mission.ground_rescue)) return null
  const event = mission.events.findLast((item) =>
    ['ground_rescue', 'summary', 'search_review', 'fleet_end', 'error'].includes(item.type))
  const last = lastFrame?.timestamp
  const timestamp = event?.timestamp
  return { timestamp: timestamp && (!last || Date.parse(timestamp) >= Date.parse(last)) ? timestamp : last }
}
