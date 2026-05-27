import { randomBytes, randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type PairingTicketStatus = "pending" | "approved" | "rejected" | "redeemed" | "expired";

export type PairingTicket = {
  id: string;
  code: string;
  remoteId: string;
  remoteSource: string;
  status: PairingTicketStatus;
  createdAt: string;
  expiresAt: string;
  resolvedAt?: string;
  resolvedBy?: string;
  redeemedAt?: string;
  redeemedSessionId?: string;
};

export type PairingStoreShape = {
  version: 1;
  tickets: PairingTicket[];
};

export class PairingTicketError extends Error {
  constructor(readonly code: "NOT_FOUND" | "INVALID_STATE", message: string) {
    super(message);
    this.name = "PairingTicketError";
  }
}

const EMPTY_STORE: PairingStoreShape = {
  version: 1,
  tickets: [],
};

const CODE_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";

export class FilePairingStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<PairingStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: PairingStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async listTickets(status?: PairingTicketStatus): Promise<PairingTicket[]> {
    const { store, changed } = await this.loadWithExpirationSweep();
    if (changed) {
      await this.save(store);
    }
    const tickets = status ? store.tickets.filter((ticket) => ticket.status === status) : store.tickets;
    return [...tickets].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }

  async getTicket(identifier: string): Promise<PairingTicket | undefined> {
    const { store, changed } = await this.loadWithExpirationSweep();
    if (changed) {
      await this.save(store);
    }
    return findTicket(store.tickets, identifier);
  }

  async createTicket(params: {
    remoteSource: string;
    remoteId?: string;
    expiresAt: string;
  }): Promise<PairingTicket> {
    const { store, changed } = await this.loadWithExpirationSweep();
    const code = generatePairingCode(new Set(store.tickets.map((ticket) => ticket.code)));
    const ticket: PairingTicket = {
      id: randomUUID(),
      code,
      remoteId: params.remoteId ?? randomUUID(),
      remoteSource: params.remoteSource,
      status: "pending",
      createdAt: new Date().toISOString(),
      expiresAt: params.expiresAt,
    };
    store.tickets.push(ticket);
    await this.save(store);
    if (changed) {
      return ticket;
    }
    return ticket;
  }

  async resolveTicket(
    identifier: string,
    params: { decision: "approved" | "rejected"; resolvedBy?: string },
  ): Promise<PairingTicket> {
    const { store, changed } = await this.loadWithExpirationSweep();
    const index = findTicketIndex(store.tickets, identifier);
    if (index < 0) {
      throw new PairingTicketError("NOT_FOUND", `unknown pairing ticket: ${identifier}`);
    }
    const ticket = store.tickets[index];
    if (!ticket || ticket.status !== "pending") {
      throw new PairingTicketError("INVALID_STATE", `pairing ticket is ${ticket?.status ?? "unknown"}: ${identifier}`);
    }
    const updated: PairingTicket = {
      ...ticket,
      status: params.decision,
      resolvedAt: new Date().toISOString(),
      ...(params.resolvedBy ? { resolvedBy: params.resolvedBy } : {}),
    };
    store.tickets[index] = updated;
    await this.save(store);
    if (changed) {
      return updated;
    }
    return updated;
  }

  async redeemTicket(code: string, params: { sessionId?: string } = {}): Promise<PairingTicket> {
    const { store, changed } = await this.loadWithExpirationSweep();
    const index = findTicketIndex(store.tickets, code);
    if (index < 0) {
      throw new PairingTicketError("NOT_FOUND", `unknown pairing ticket: ${code}`);
    }
    const ticket = store.tickets[index];
    if (!ticket || ticket.status !== "approved") {
      throw new PairingTicketError("INVALID_STATE", `pairing ticket is ${ticket?.status ?? "unknown"}: ${code}`);
    }
    const updated: PairingTicket = {
      ...ticket,
      status: "redeemed",
      redeemedAt: new Date().toISOString(),
      ...(params.sessionId ? { redeemedSessionId: params.sessionId } : {}),
    };
    store.tickets[index] = updated;
    await this.save(store);
    if (changed) {
      return updated;
    }
    return updated;
  }

  private async loadWithExpirationSweep(): Promise<{ store: PairingStoreShape; changed: boolean }> {
    const store = await this.load();
    const changed = expireTickets(store.tickets, new Date().toISOString());
    return { store, changed };
  }
}

function findTicket(tickets: PairingTicket[], identifier: string): PairingTicket | undefined {
  const normalized = normalizeIdentifier(identifier);
  return tickets.find((ticket) => ticket.id === identifier || ticket.code === normalized);
}

function findTicketIndex(tickets: PairingTicket[], identifier: string): number {
  const normalized = normalizeIdentifier(identifier);
  return tickets.findIndex((ticket) => ticket.id === identifier || ticket.code === normalized);
}

function normalizeIdentifier(identifier: string): string {
  return identifier.trim().toUpperCase();
}

function expireTickets(tickets: PairingTicket[], nowIso: string): boolean {
  let changed = false;
  for (const ticket of tickets) {
    if ((ticket.status === "pending" || ticket.status === "approved") && ticket.expiresAt <= nowIso) {
      ticket.status = "expired";
      ticket.resolvedAt ??= nowIso;
      ticket.resolvedBy ??= "system";
      changed = true;
    }
  }
  return changed;
}

function generatePairingCode(existingCodes: Set<string>): string {
  while (true) {
    const buffer = randomBytes(8);
    let code = "";
    for (const byte of buffer) {
      code += CODE_ALPHABET[byte % CODE_ALPHABET.length];
    }
    const formatted = `${code.slice(0, 4)}-${code.slice(4, 8)}`;
    if (!existingCodes.has(formatted)) {
      return formatted;
    }
  }
}
