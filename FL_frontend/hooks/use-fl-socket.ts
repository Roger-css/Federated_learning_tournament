'use client'

import { useEffect, useRef, useState } from 'react'
import * as signalR from '@microsoft/signalr'
import { FL_HUB_URL, type ClientMetrics, type RoundResult } from '@/lib/fl-constants'

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error'

export interface FlSocketData {
  localBaseline: ClientMetrics[]
  rounds: RoundResult[]
  connectionState: ConnectionState
}

export function useFlSocket(): FlSocketData {
  const [localBaseline, setLocalBaseline] = useState<ClientMetrics[]>([])
  const [rounds, setRounds] = useState<RoundResult[]>([])
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')
  const connectionRef = useRef<signalR.HubConnection | null>(null)

  useEffect(() => {
    const connection = new signalR.HubConnectionBuilder()
      .withUrl(FL_HUB_URL)
      .withAutomaticReconnect()
      .configureLogging(signalR.LogLevel.Warning)
      .build()

    connectionRef.current = connection

    connection.on('LocalBaselineUpdated', (data: ClientMetrics[]) => {
      setLocalBaseline(data)
    })

    connection.on('RoundUpdated', (round: RoundResult) => {
      setRounds((prev) => {
        // Replace if this round number already exists, otherwise append
        const exists = prev.some((r) => r.roundNumber === round.roundNumber)
        if (exists) {
          return prev.map((r) => (r.roundNumber === round.roundNumber ? round : r))
        }
        return [...prev, round].sort((a, b) => a.roundNumber - b.roundNumber)
      })
    })

    connection.onreconnecting(() => setConnectionState('connecting'))
    connection.onreconnected(() => setConnectionState('connected'))
    connection.onclose(() => setConnectionState('disconnected'))

    connection
      .start()
      .then(() => setConnectionState('connected'))
      .catch(() => setConnectionState('error'))

    return () => {
      connection.stop()
    }
  }, [])

  return { localBaseline, rounds, connectionState }
}
