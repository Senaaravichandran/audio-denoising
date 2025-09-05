import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, jsonb, integer, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const audioJobs = pgTable("audio_jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id"),
  filename: text("filename").notNull(),
  originalFormat: text("original_format").notNull(),
  outputFormat: text("output_format").notNull(),
  fileSize: integer("file_size").notNull(),
  status: text("status").notNull().default("pending"), // pending, processing, completed, failed
  processingOptions: jsonb("processing_options"),
  originalPath: text("original_path"),
  processedPath: text("processed_path"),
  noiseReductionLevel: integer("noise_reduction_level").default(7),
  voicePreservation: integer("voice_preservation").default(9),
  processingMode: text("processing_mode").default("balanced"),
  progress: integer("progress").default(0),
  errorMessage: text("error_message"),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  createdAt: timestamp("created_at").defaultNow(),
  // Video processing fields
  isVideo: boolean("is_video").default(false),
  enhancedAudioPath: text("enhanced_audio_path"),
  stage: text("stage").default("upload"), // upload, video_extraction, ai_denoising, video_combining, completed
  aiExplanation: text("ai_explanation"), // AI-generated explanation of the processing
});

export const noiseSamples = pgTable("noise_samples", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description"),
  filePath: text("file_path").notNull(),
  noiseType: text("noise_type").notNull(), // traffic, fan, typing, etc.
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertAudioJobSchema = createInsertSchema(audioJobs).omit({
  id: true,
  createdAt: true,
}).extend({
  processingOptions: z.record(z.any()).optional(),
});

export const insertNoiseSampleSchema = createInsertSchema(noiseSamples).omit({
  id: true,
  createdAt: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type InsertAudioJob = z.infer<typeof insertAudioJobSchema>;
export type AudioJob = typeof audioJobs.$inferSelect;
export type InsertNoiseSample = z.infer<typeof insertNoiseSampleSchema>;
export type NoiseSample = typeof noiseSamples.$inferSelect;
