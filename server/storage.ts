import { type User, type InsertUser, type AudioJob, type InsertAudioJob, type NoiseSample, type InsertNoiseSample } from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  // Audio Jobs
  createAudioJob(job: InsertAudioJob): Promise<AudioJob>;
  getAudioJob(id: string): Promise<AudioJob | undefined>;
  updateAudioJob(id: string, updates: Partial<AudioJob>): Promise<AudioJob | undefined>;
  listAudioJobs(userId?: string): Promise<AudioJob[]>;
  deleteAudioJob(id: string): Promise<boolean>;
  
  // Noise Samples
  createNoiseSample(sample: InsertNoiseSample): Promise<NoiseSample>;
  getNoiseSample(id: string): Promise<NoiseSample | undefined>;
  listNoiseSamples(noiseType?: string): Promise<NoiseSample[]>;
  deleteNoiseSample(id: string): Promise<boolean>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;
  private audioJobs: Map<string, AudioJob>;
  private noiseSamples: Map<string, NoiseSample>;

  constructor() {
    this.users = new Map();
    this.audioJobs = new Map();
    this.noiseSamples = new Map();
  }

  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async createAudioJob(insertJob: InsertAudioJob): Promise<AudioJob> {
    const id = randomUUID();
    const job: AudioJob = {
      ...insertJob,
      id,
      createdAt: new Date(),
      startedAt: null,
      completedAt: null,
    };
    this.audioJobs.set(id, job);
    return job;
  }

  async getAudioJob(id: string): Promise<AudioJob | undefined> {
    return this.audioJobs.get(id);
  }

  async updateAudioJob(id: string, updates: Partial<AudioJob>): Promise<AudioJob | undefined> {
    const job = this.audioJobs.get(id);
    if (!job) return undefined;
    
    const updatedJob = { ...job, ...updates };
    this.audioJobs.set(id, updatedJob);
    return updatedJob;
  }

  async listAudioJobs(userId?: string): Promise<AudioJob[]> {
    const jobs = Array.from(this.audioJobs.values());
    if (userId) {
      return jobs.filter(job => job.userId === userId);
    }
    return jobs;
  }

  async deleteAudioJob(id: string): Promise<boolean> {
    return this.audioJobs.delete(id);
  }

  async createNoiseSample(insertSample: InsertNoiseSample): Promise<NoiseSample> {
    const id = randomUUID();
    const sample: NoiseSample = {
      ...insertSample,
      id,
      createdAt: new Date(),
    };
    this.noiseSamples.set(id, sample);
    return sample;
  }

  async getNoiseSample(id: string): Promise<NoiseSample | undefined> {
    return this.noiseSamples.get(id);
  }

  async listNoiseSamples(noiseType?: string): Promise<NoiseSample[]> {
    const samples = Array.from(this.noiseSamples.values());
    if (noiseType) {
      return samples.filter(sample => sample.noiseType === noiseType);
    }
    return samples;
  }

  async deleteNoiseSample(id: string): Promise<boolean> {
    return this.noiseSamples.delete(id);
  }
}

export const storage = new MemStorage();
